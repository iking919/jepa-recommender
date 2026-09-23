import torch
import os

def save_checkpoint(state: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath: str, map_location=None):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")
    return torch.load(filepath, map_location=map_location, weights_only=False)