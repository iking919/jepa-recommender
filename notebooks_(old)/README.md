# Notebooks

This directory contains Colab notebooks used for exploratory data analysis, experimentation, visualization, and development.

## Intended Use

Notebooks may be used to:

- Explore the MovieLens-1M dataset.
- Inspect user interaction sequences and item distributions.
- Test preprocessing and modeling ideas.
- Visualize training behavior and evaluation metrics.
- Investigate experimental results.

## Relationship to the Source Code

Reusable functionality should be implemented in the `src/` package rather than duplicated across notebooks.

Where possible, notebooks should import project modules so that exploratory experiments remain consistent with the main training and evaluation pipelines.

## Reproducibility

For notebooks that produce results used in reports or research analysis:

- Record the configuration and dataset version.
- Set random seeds where applicable.
- Document any manual preprocessing or filtering.
- Distinguish exploratory results from final held-out test results.
- Avoid relying on unrecorded state from previously executed cells.

Notebook outputs and checkpoints should not be treated as substitutes for the reproducible training and evaluation scripts.