# Dataset

This directory contains the data used for the sequential recommendation experiments in this project.

The primary dataset is **MovieLens-1M**, provided by the GroupLens Research project at the University of Minnesota.

## Directory Structure

```text
data/
├── README.md
├── raw/
│   └── ml-1m/
│       ├── README
│       ├── movies.dat
│       ├── ratings.dat
│       └── users.dat
│
├── interim/
│   └── ...
│
└── processed/
    └── ...
```

### `raw/`

Contains the original MovieLens-1M files after downloading and extracting the official dataset archive.

The raw data should not be modified.

### `interim/`

Contains intermediate datasets generated during preprocessing, filtering, sequence construction, and item/user mapping.

These files are derived from the raw dataset and can be regenerated using the preprocessing scripts.

### `processed/`

Contains the final datasets used by the BERT4Rec and JEPA experiments.

These files are generated from the raw MovieLens-1M data and are not stored in the repository.

---

## MovieLens-1M

MovieLens-1M contains approximately one million movie ratings collected from users of the MovieLens recommendation service.

The original dataset contains:

* **1,000,209 ratings**
* **6,040 users**
* **3,706 movies**
* Ratings on a **1–5 star scale**
* Timestamped user-item interactions

The dataset consists of three primary files:

| File          | Description                            |
| ------------- | -------------------------------------- |
| `ratings.dat` | User ratings of movies with timestamps |
| `users.dat`   | User demographic information           |
| `movies.dat`  | Movie metadata and genres              |

For this project, `ratings.dat` is the primary source used to construct chronological user interaction sequences. Movie IDs are used as item identifiers for sequential recommendation.

---

## Downloading the Dataset

The dataset should be downloaded using:

```bash
python scripts/download_ml1m.py
```

The script downloads and extracts the dataset into:

```text
data/raw/ml-1m/
```

After extraction, the expected structure is:

```text
data/raw/ml-1m/
├── README
├── movies.dat
├── ratings.dat
└── users.dat
```

The raw dataset should remain unchanged so that all preprocessing steps can be reproduced from the original data.

---

## Preprocessing

After downloading MovieLens-1M, run:

```bash
python scripts/preprocess_ml1m.py
```

The preprocessing pipeline performs the transformations required by the sequential recommendation experiments.

The current experimental preprocessing includes:

1. Loading the MovieLens-1M ratings.
2. Ordering interactions chronologically for each user.
3. Filtering movies according to the experimental minimum-interaction threshold.
4. Removing users/interactions that do not satisfy the required sequence constraints.
5. Mapping MovieLens movie IDs to contiguous model item IDs.
6. Constructing chronological user interaction sequences.
7. Limiting sequences to the configured maximum sequence length.
8. Creating the train/validation/test splits.
9. Constructing training examples for BERT4Rec.
10. Constructing context-target training pairs for JEPA.

The exact preprocessing configuration should be specified in the corresponding experiment configuration files.

---

## Current Experimental Dataset

The current MovieLens-1M experiments use a minimum movie-interaction threshold of **5 ratings per movie** and a maximum sequence length of **200**.

After preprocessing, the common experimental dataset contains:

* **6,040 users**
* **3,416 mapped movies**
* **3,418 vocabulary entries**, including special tokens
* Chronological user sequences
* Maximum sequence length: **200**

The resulting training data differs slightly between the two models because BERT4Rec and JEPA use different training-example construction procedures.

### BERT4Rec

BERT4Rec uses masked sequential prediction examples constructed from chronological user histories.

### JEPA

JEPA uses context-target pairs in which a context sequence is used to predict a target item representation.

The current JEPA training configuration contains approximately **642,719 context-target training pairs**.

---

## Reproducibility

The raw and processed datasets are intentionally excluded from version control.

To reproduce the dataset from a clean checkout:

```bash
python scripts/download_ml1m.py
python scripts/preprocess_ml1m.py
```

This allows the complete dataset pipeline to be regenerated from the original MovieLens-1M data.

Experiment-specific preprocessing parameters should be stored in the corresponding configuration files under:

```text
configs/
```

---

## Data Versioning

The repository does not redistribute MovieLens-1M.

Users should obtain the dataset from the official MovieLens/GroupLens distribution and follow the dataset's applicable terms of use.

The preprocessing code in this repository is provided to reproduce the datasets used in the experiments.

---

## Citation

If using MovieLens-1M, please cite the original MovieLens dataset publication:

> F. M. Harper and J. A. Konstan, "The MovieLens Datasets: History and Context," ACM Transactions on Interactive Intelligent Systems, vol. 5, no. 4, pp. 1–19, 2015.

See the official GroupLens MovieLens documentation for dataset information and usage terms.
