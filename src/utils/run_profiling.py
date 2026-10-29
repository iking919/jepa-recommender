"""
Shared profiling utilities and the uniform run-summary schema.

Every training script (JEPA, causal JEPA, BERT4Rec) should:

1. create a ``MemoryTracker`` before building anything,
2. wrap setup / train / validation in ``tracker.phase(...)``,
3. call ``tracker.end_epoch(...)`` once per epoch and merge the result into
   the epoch history record,
4. call ``build_run_summary(...)`` at the end.

Because all scripts use the same builder, the resulting ``run_summary.json``
files have identical keys and can be compared directly.

All memory values are reported in GiB (1024**3 bytes) unless a key ends in
``_bytes``.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

import torch

GIB = 1024 ** 3
SCHEMA_VERSION = "1.0"

METRIC_NAMES = ("HR@5", "HR@10", "NDCG@5", "NDCG@10")

# Keys every model must report under "training_configuration". Missing keys
# are filled with None so the schema stays identical across models.
COMMON_TRAINING_KEYS = (
    "batch_size",
    "learning_rate",
    "weight_decay",
    "mixed_precision",
    "amp_dtype",
    "compile",
    "patience",
    "lr_schedule",
    "warmup_epochs",
    "min_lr_ratio",
    "selection_window",
    "save_checkpoint",
)


# ---------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------

def _gib(num_bytes: float | int | None) -> float:
    return (num_bytes or 0) / GIB


def cpu_rss_bytes() -> int:
    """Current resident set size of this process."""
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        pass
    try:
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return 0


def cpu_peak_rss_bytes() -> int:
    """Peak resident set size of this process since it started."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on macOS and kilobytes on Linux.
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, ValueError):
        return cpu_rss_bytes()


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def module_memory(module: torch.nn.Module | None) -> dict[str, int]:
    """Parameter / buffer footprint of a module, in bytes."""
    if module is None:
        return {"parameters_bytes": 0, "trainable_bytes": 0, "buffers_bytes": 0}

    parameters = list(module.parameters())
    return {
        "parameters_bytes": sum(tensor_bytes(p) for p in parameters),
        "trainable_bytes": sum(
            tensor_bytes(p) for p in parameters if p.requires_grad
        ),
        "buffers_bytes": sum(tensor_bytes(b) for b in module.buffers()),
    }


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Total size of tensors held in the optimizer state."""
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += tensor_bytes(value)
    return total


def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------
# Memory tracker
# ---------------------------------------------------------------------

class MemoryTracker:
    """Tracks GPU and CPU memory across named phases and epochs."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cuda = device.type == "cuda"
        self.checkpoints: dict[str, dict[str, int]] = {}
        self.phase_stats: dict[str, dict[str, float]] = {}
        self.epochs: list[dict[str, float]] = []
        self._last_phase: dict[str, dict[str, float]] = {}
        self._cpu_peak_seen = 0

        if self.cuda:
            torch.cuda.reset_peak_memory_stats(device)

    # -- internals -----------------------------------------------------

    def _sync(self) -> None:
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def _churn_bytes(self) -> int:
        """Cumulative bytes ever allocated by the CUDA caching allocator."""
        if not self.cuda:
            return 0
        stats = torch.cuda.memory_stats(self.device)
        return int(stats.get("allocated_bytes.all.allocated", 0))

    def _allocated(self) -> int:
        return torch.cuda.memory_allocated(self.device) if self.cuda else 0

    def _reserved(self) -> int:
        return torch.cuda.memory_reserved(self.device) if self.cuda else 0

    # -- public API ----------------------------------------------------

    def mark(self, label: str) -> dict[str, int]:
        """Record current (not peak) memory under a label."""
        self._sync()
        rss = cpu_rss_bytes()
        self._cpu_peak_seen = max(self._cpu_peak_seen, rss)
        snapshot = {
            "gpu_allocated_bytes": self._allocated(),
            "gpu_reserved_bytes": self._reserved(),
            "cpu_rss_bytes": rss,
        }
        self.checkpoints[label] = snapshot
        return snapshot

    def delta_allocated_bytes(self, start: str, end: str) -> int:
        """GPU bytes allocated between two marks."""
        return (
            self.checkpoints[end]["gpu_allocated_bytes"]
            - self.checkpoints[start]["gpu_allocated_bytes"]
        )

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Measure peak GPU memory inside the block, e.g. 'train'."""
        self._sync()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        churn_start = self._churn_bytes()
        start = time.perf_counter()

        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - start
            rss = cpu_rss_bytes()
            self._cpu_peak_seen = max(self._cpu_peak_seen, rss)

            peak_allocated = (
                torch.cuda.max_memory_allocated(self.device)
                if self.cuda else 0
            )
            peak_reserved = (
                torch.cuda.max_memory_reserved(self.device)
                if self.cuda else 0
            )
            churn = self._churn_bytes() - churn_start

            self._last_phase[name] = {
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "end_allocated_bytes": self._allocated(),
                "end_reserved_bytes": self._reserved(),
                "churn_bytes": churn,
                "cpu_rss_bytes": rss,
                "seconds": elapsed,
            }

            agg = self.phase_stats.setdefault(
                name,
                {
                    "peak_allocated_bytes": 0,
                    "peak_reserved_bytes": 0,
                    "churn_bytes": 0,
                    "seconds": 0.0,
                    "calls": 0,
                },
            )
            agg["peak_allocated_bytes"] = max(
                agg["peak_allocated_bytes"], peak_allocated
            )
            agg["peak_reserved_bytes"] = max(
                agg["peak_reserved_bytes"], peak_reserved
            )
            agg["churn_bytes"] += churn
            agg["seconds"] += elapsed
            agg["calls"] += 1

    def end_epoch(
        self,
        epoch: int,
        train_seconds: float,
        validation_seconds: float,
    ) -> dict[str, float]:
        """Flat per-epoch memory record for the history CSV.

        Must be called after the 'train' and 'validation' phases of the
        epoch have both completed.
        """
        train = self._last_phase.get("train", {})
        val = self._last_phase.get("validation", {})

        def get(phase: dict, key: str) -> float:
            return phase.get(key, 0)

        epoch_peak_allocated = max(
            get(train, "peak_allocated_bytes"),
            get(val, "peak_allocated_bytes"),
        )
        epoch_peak_reserved = max(
            get(train, "peak_reserved_bytes"),
            get(val, "peak_reserved_bytes"),
        )

        record = {
            "gpu_train_peak_allocated_gb": _gib(
                get(train, "peak_allocated_bytes")
            ),
            "gpu_train_peak_reserved_gb": _gib(
                get(train, "peak_reserved_bytes")
            ),
            "gpu_val_peak_allocated_gb": _gib(
                get(val, "peak_allocated_bytes")
            ),
            "gpu_val_peak_reserved_gb": _gib(
                get(val, "peak_reserved_bytes")
            ),
            "gpu_epoch_peak_allocated_gb": _gib(epoch_peak_allocated),
            "gpu_epoch_peak_reserved_gb": _gib(epoch_peak_reserved),
            "gpu_allocated_after_epoch_gb": _gib(
                get(val, "end_allocated_bytes")
            ),
            "gpu_alloc_churn_gb": _gib(
                get(train, "churn_bytes") + get(val, "churn_bytes")
            ),
            "cpu_rss_gb": _gib(cpu_rss_bytes()),
            # Kept for backwards compatibility with older histories.
            "peak_gpu_memory_gb": _gib(epoch_peak_allocated),
            # Internal fields used to integrate memory over time.
            "_epoch_seconds": train_seconds + validation_seconds,
            "_epoch_peak_reserved_bytes": epoch_peak_reserved,
        }
        self.epochs.append({"epoch": epoch, **record})

        # Hide internal fields from the public history record.
        return {k: v for k, v in record.items() if not k.startswith("_")}

    # -- summary -------------------------------------------------------

    def summary(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        num_parameters: int,
    ) -> dict[str, Any]:
        """Aggregate every measurement into the 'memory' summary block."""

        # ---- static footprint ----
        total_mem = module_memory(model)
        context_mem = module_memory(getattr(model, "context_encoder", None))
        target_mem = module_memory(getattr(model, "target_encoder", None))
        opt_bytes = optimizer_state_bytes(optimizer)

        data_resident_bytes = 0
        if "process_start" in self.checkpoints and "after_data" in self.checkpoints:
            data_resident_bytes = max(
                self.delta_allocated_bytes("process_start", "after_data"), 0
            )

        static_resident_bytes = (
            total_mem["parameters_bytes"]
            + total_mem["buffers_bytes"]
            + opt_bytes
            + data_resident_bytes
        )

        static = {
            "parameters_gb": _gib(total_mem["parameters_bytes"]),
            "trainable_parameters_gb": _gib(total_mem["trainable_bytes"]),
            "buffers_gb": _gib(total_mem["buffers_bytes"]),
            "gradients_gb_estimate": _gib(total_mem["trainable_bytes"]),
            "optimizer_state_gb": _gib(opt_bytes),
            "context_encoder_gb": _gib(context_mem["parameters_bytes"]),
            "target_encoder_gb": _gib(target_mem["parameters_bytes"]),
            "data_resident_on_gpu_gb": _gib(data_resident_bytes),
            "static_resident_gb": _gib(static_resident_bytes),
            "checkpoints": {
                label: {
                    "gpu_allocated_gb": _gib(snap["gpu_allocated_bytes"]),
                    "gpu_reserved_gb": _gib(snap["gpu_reserved_bytes"]),
                    "cpu_rss_gb": _gib(snap["cpu_rss_bytes"]),
                }
                for label, snap in self.checkpoints.items()
            },
        }

        # ---- per-phase peaks ----
        peak_by_phase = {
            name: {
                "peak_allocated_gb": _gib(stats["peak_allocated_bytes"]),
                "peak_reserved_gb": _gib(stats["peak_reserved_bytes"]),
                "cumulative_alloc_churn_gb": _gib(stats["churn_bytes"]),
                "total_seconds": stats["seconds"],
                "calls": int(stats["calls"]),
            }
            for name, stats in self.phase_stats.items()
        }

        overall_peak_allocated = max(
            (s["peak_allocated_bytes"] for s in self.phase_stats.values()),
            default=0,
        )
        overall_peak_reserved = max(
            (s["peak_reserved_bytes"] for s in self.phase_stats.values()),
            default=0,
        )
        total_churn = sum(
            s["churn_bytes"] for s in self.phase_stats.values()
        )

        # ---- per-epoch statistics ----
        def stat_block(values: list[float]) -> dict[str, float]:
            if not values:
                return {"max": 0.0, "mean": 0.0, "min": 0.0}
            return {
                "max": max(values),
                "mean": sum(values) / len(values),
                "min": min(values),
            }

        epoch_peak_allocated = stat_block(
            [e["gpu_epoch_peak_allocated_gb"] for e in self.epochs]
        )
        epoch_peak_reserved = stat_block(
            [e["gpu_epoch_peak_reserved_gb"] for e in self.epochs]
        )

        # Reserved memory integrated over wall-clock time (GiB-seconds).
        gb_seconds_reserved = sum(
            _gib(e["_epoch_peak_reserved_bytes"]) * e["_epoch_seconds"]
            for e in self.epochs
        )

        # ---- device / allocator ----
        total_capacity = 0
        allocator = {"num_alloc_retries": 0, "num_ooms": 0}
        device_name = None

        if self.cuda:
            total_capacity = torch.cuda.get_device_properties(
                self.device
            ).total_memory
            device_name = torch.cuda.get_device_name(self.device)
            stats = torch.cuda.memory_stats(self.device)
            allocator = {
                "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
                "num_ooms": int(stats.get("num_ooms", 0)),
            }

        train_peak = self.phase_stats.get("train", {}).get(
            "peak_allocated_bytes", 0
        )
        dynamic_overhead = max(train_peak - static_resident_bytes, 0)

        gpu = {
            "available": self.cuda,
            "device_name": device_name,
            "device_total_capacity_gb": _gib(total_capacity),
            "peak_allocated_gb": _gib(overall_peak_allocated),
            "peak_reserved_gb": _gib(overall_peak_reserved),
            "peak_utilization_fraction": (
                overall_peak_reserved / total_capacity
                if total_capacity else 0.0
            ),
            "final_allocated_gb": _gib(self._allocated()),
            "final_reserved_gb": _gib(self._reserved()),
            "epoch_peak_allocated_gb": epoch_peak_allocated,
            "epoch_peak_reserved_gb": epoch_peak_reserved,
            "peak_by_phase": peak_by_phase,
            "cumulative_alloc_churn_gb": _gib(total_churn),
            "gb_seconds_reserved": gb_seconds_reserved,
            "allocator": allocator,
        }

        cpu = {
            "peak_rss_gb": _gib(
                max(cpu_peak_rss_bytes(), self._cpu_peak_seen)
            ),
            "final_rss_gb": _gib(cpu_rss_bytes()),
            "start_rss_gb": _gib(
                self.checkpoints.get("process_start", {}).get(
                    "cpu_rss_bytes", 0
                )
            ),
        }

        efficiency = {
            "peak_allocated_bytes_per_parameter": (
                overall_peak_allocated / num_parameters
                if num_parameters else 0.0
            ),
            "peak_reserved_bytes_per_parameter": (
                overall_peak_reserved / num_parameters
                if num_parameters else 0.0
            ),
            "peak_allocated_gb_per_million_parameters": (
                _gib(overall_peak_allocated) / (num_parameters / 1e6)
                if num_parameters else 0.0
            ),
            "activation_and_gradient_overhead_gb": _gib(dynamic_overhead),
            "static_fraction_of_train_peak": (
                static_resident_bytes / train_peak if train_peak else 0.0
            ),
        }

        return {
            "units": "GiB (1024^3 bytes) unless key ends in _bytes",
            "gpu": gpu,
            "cpu": cpu,
            "static": static,
            "efficiency": efficiency,
        }


# ---------------------------------------------------------------------
# Uniform summary builder
# ---------------------------------------------------------------------

def build_run_summary(
    *,
    model_name: str,
    experiment_name: str,
    dataset: str,
    device: torch.device,
    seed: int,
    run_start: str,
    run_end: str,
    total_wall_seconds: float,
    configured_epochs: int,
    epoch_history: list[dict],
    best_epoch: int,
    best_metrics: dict,
    best_epoch_loss: float | None,
    best_selection_score: float | None,
    early_stopped: bool,
    training_configuration: dict[str, Any],
    model_configuration: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tracker: MemoryTracker,
    data_info: dict[str, Any],
    samples_per_epoch: int,
    files: dict[str, str | None],
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the run summary with an identical schema for every model."""

    completed = len(epoch_history)
    total_params, trainable_params = count_parameters(model)

    # ---- timing ----
    train_total = sum(r["train_time_seconds"] for r in epoch_history)
    validation_total = sum(r["validation_time_seconds"] for r in epoch_history)
    epoch_total = train_total + validation_total
    time_to_best = sum(
        r["epoch_time_seconds"] for r in epoch_history[:best_epoch]
    )

    def mean(total: float) -> float:
        return total / completed if completed else 0.0

    timing = {
        "total_wall_clock_seconds": total_wall_seconds,
        "total_wall_clock_minutes": total_wall_seconds / 60.0,
        "total_train_seconds": train_total,
        "total_validation_seconds": validation_total,
        "total_epoch_seconds": epoch_total,
        "setup_and_overhead_seconds": max(
            total_wall_seconds - epoch_total, 0.0
        ),
        "time_to_best_epoch_seconds": time_to_best,
        "average_epoch_seconds": mean(epoch_total),
        "average_train_seconds": mean(train_total),
        "average_validation_seconds": mean(validation_total),
        "train_samples_per_second": (
            samples_per_epoch * completed / train_total
            if train_total else 0.0
        ),
    }

    # ---- parameters ----
    parameters = {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }

    # ---- training configuration with a fixed set of common keys ----
    train_cfg = {key: None for key in COMMON_TRAINING_KEYS}
    train_cfg.update(training_configuration)

    # ---- memory ----
    memory = tracker.summary(
        model=model,
        optimizer=optimizer,
        num_parameters=total_params,
    )
    memory["gpu"]["peak_allocated_gb_per_epoch_history"] = [
        r.get("gpu_epoch_peak_allocated_gb", 0.0) for r in epoch_history
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "model": model_name,
        "experiment_name": experiment_name,
        "dataset": dataset,
        "run": {
            "start": run_start,
            "end": run_end,
            "device": str(device),
            "seed": seed,
        },
        "epochs": {
            "configured": configured_epochs,
            "completed": completed,
            "best_epoch": best_epoch,
            "early_stopped": early_stopped,
        },
        "best": {
            "epoch": best_epoch,
            "metrics": {
                name: best_metrics.get(name, 0.0) for name in METRIC_NAMES
            },
            "loss": best_epoch_loss,
            "selection_score": best_selection_score,
        },
        "timing": timing,
        "memory": memory,
        "parameters": parameters,
        "training_configuration": train_cfg,
        "model_configuration": model_configuration,
        "data": data_info,
        "files": files,
        "extras": extras or {},
    }
