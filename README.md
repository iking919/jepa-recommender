# Newman-Inspired Movie Ranking for Sequential Recommendation

A research project for **CSCI 6250: Frontiers of Network Science** at Rensselaer Polytechnic Institute.

This project investigates whether a global movie ranking inferred from heterogeneous pairwise comparisons can improve personalized sequential recommendations produced by BERT4Rec. It builds on M. E. J. Newman's *Ranking with Multiple Types of Pairwise Comparisons* and extends the general idea of heterogeneous comparison-based ranking to movie recommendation.

**Project status:** Experimental design and baseline preparation.

## Research Question

Can a global movie ranking inferred from multiple types of pairwise comparisons improve the top-10 recommendation performance of BERT4Rec on MovieLens-1M?

The primary hypothesis is that incorporating information from a global movie comparison network will improve NDCG@10 compared with BERT4Rec alone. HR@10 will be evaluated as a secondary ranking metric.

The project will also investigate whether any improvement is attributable to the pairwise-ranking method or can be explained by a simpler popularity-based reranking baseline.

## Background

Sequential recommendation models learn from a user's interaction history to predict items they may interact with next. BERT4Rec uses a bidirectional Transformer and masked item prediction to learn representations of user interaction sequences.

Newman's ranking framework addresses a different problem: inferring relative rankings from pairwise comparisons that can belong to different interaction types. Rather than treating every comparison as equivalent, the method accounts for differences between types of interactions.

This project investigates whether a global ranking inferred from movie comparisons can provide complementary information to a personalized sequential recommendation model.

## Research Objectives

1. Establish a reproducible BERT4Rec baseline using MovieLens-1M.
2. Construct a movie comparison network using training-set ratings.
3. Adapt Newman's multiple-comparison ranking approach to infer global movie scores.
4. Integrate the global ranking with BERT4Rec recommendation scores.
5. Compare the combined model with BERT4Rec alone and a popularity-based reranking baseline.
6. Evaluate recommendation quality and computational overhead.
7. Analyze the conditions under which global ranking information helps or hurts personalized recommendation.

## Dataset

The primary dataset is **MovieLens-1M**, published by the GroupLens Research Project at the University of Minnesota.

The original dataset contains:

| Statistic | Count |
|---|---:|
| Ratings | 1,000,209 |
| Users | 6,040 |
| Movies | 3,706 |

The dataset provides user IDs, movie IDs, explicit ratings, and timestamps.

The existing BERT4Rec research pipeline filters movies with fewer than five interactions and constructs chronological user interaction sequences. This project will reuse the established preprocessing and split definitions to preserve comparability with the existing baseline.

The processed data used in the current recommender experiments contains 6,040 users and 3,416 mapped movies. These counts will be verified against the artifacts used in the course-project baseline.

Original dataset files and derived datasets will not be committed to the repository.

## Experimental Design

### 1. BERT4Rec Baseline

BERT4Rec serves as the primary recommendation baseline. It learns personalized rankings from user interaction sequences using masked item prediction.

The baseline will use the established tuned configuration and the same chronological training, validation, and test split definitions as the existing recommender research.

The baseline checkpoint and configuration will be recorded so that the experiment can be reproduced.

### 2. Movie Comparison Network

Movies will be represented as nodes. Pairwise comparisons will be derived from ratings assigned by the same user to different movies.

When a user rates one movie higher than another, the higher-rated movie will be treated as the winner of that comparison. Equal-rated pairs will initially be excluded.

The initial comparison types will be defined by absolute rating differences:

- **Close comparisons:** rating differences of one star.
- **Large-difference comparisons:** rating differences of two or more stars.

These categories are a proposed adaptation for MovieLens-1M and will be documented as such.

Only training-set ratings will be used to construct the network. Validation and test ratings will not contribute to the inferred global ranking.

The ranking implementation will follow the relevant model assumptions and inference procedure in Newman's paper as closely as appropriate for the adapted data. Any changes required for the movie domain will be documented.

### 3. Global Movie Ranking

The comparison network will be used to estimate a global score for each movie. This score will represent its inferred relative standing across the comparison data.

The implementation will record the comparison types, preprocessing choices, ranking procedure, and treatment of movies with limited comparison information.

The resulting scores will be inspected before being incorporated into the sequential recommendation model.

### 4. Score Fusion

The global movie score will be combined with BERT4Rec's personalized recommendation score:

s_combined(u, i) = s_BERT4Rec(u, i) + lambda * g(i)

where:

- u denotes a user.
- i denotes a candidate movie.
- s_BERT4Rec(u, i) is the original BERT4Rec score.
- g(i) is the normalized global movie-ranking score.
- lambda controls the contribution of the global ranking.

The score normalization procedure and candidate-item handling will be held consistent across experiments.

The fusion weight will be selected using validation performance, primarily NDCG@10. The test set will be reserved for final evaluation after model selection.

## Experiments

The following experiments will be conducted:

| Experiment | Description |
|---|---|
| A | BERT4Rec baseline |
| B | Global comparison-network ranking alone |
| C | BERT4Rec combined with the global ranking |
| D | BERT4Rec combined with different global-ranking weights |
| E | BERT4Rec combined with a popularity-based ranking signal |

The popularity-based comparison is intended to determine whether the network method contributes useful information beyond a simpler global ranking heuristic.

## Evaluation

The primary evaluation metrics are:

- **HR@10:** Measures whether the held-out target movie appears among the ten highest-ranked recommendations.
- **NDCG@10:** Measures ranking quality while assigning greater weight to relevant movies appearing closer to the top of the recommendation list.

HR@5 and NDCG@5 may also be reported as supplementary metrics.

Evaluation will use the same chronological splits, candidate-item handling, and exclusion of previously interacted-with movies across the compared recommendation systems.

Validation performance will be used for model selection. The held-out test set will be used for final reporting only after the model and fusion weight have been selected.

Computational measurements will include network construction time, ranking computation time, recommendation evaluation time, and additional memory requirements where available.

## Determining Success

The primary success criterion is an improvement in validation NDCG@10 over the BERT4Rec baseline, followed by confirmation on the held-out test set.

HR@10 will provide complementary evidence about whether the combined system retrieves the target movie more frequently.

Results will be interpreted alongside the popularity-based baseline and computational overhead.

Possible outcomes include:

- **Improvement:** The network ranking provides complementary information that improves sequential recommendation.
- **No meaningful change:** The global ranking may contain little additional information beyond BERT4Rec.
- **Degradation:** The global ranking may interfere with personalized recommendations.
- **Popularity-equivalent improvement:** The gain may be attributable to global item popularity rather than the heterogeneous comparison method.

All outcomes will be reported, including negative results.

## Repository Structure

```text
newman-movie-reranking/
├── configs/
│   └── bert4rec_baseline.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── checkpoints/
├── results/
│   ├── bert4rec_baseline/
│   ├── network_ranking/
│   └── score_fusion/
├── scripts/
│   ├── build_movie_network.py
│   ├── run_network_ranking.py
│   ├── run_score_fusion.py
│   └── run_evaluation.py
├── src/
│   ├── models/
│   ├── data/
│   ├── network/
│   ├── ranking/
│   ├── recommendation/
│   └── evaluation/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

This structure describes the intended project organization. Script names and module paths will be updated to match the actual implementation.

## Reproducibility

The project will preserve:

1. The BERT4Rec configuration and checkpoint identifier.
2. The preprocessing and chronological split definitions.
3. The movie comparison types and network construction parameters.
4. The ranking implementation and score-normalization procedure.
5. The fusion weights and validation-based selection criterion.
6. Individual experiment results and aggregate evaluation tables.
7. The software environment and hardware information.
8. The final held-out test results.

Random seeds will be recorded for experiments involving stochastic model training or randomized procedures.

## Relationship to Existing Research

This repository is a focused extension of the broader JEPA and BERT4Rec sequential recommendation research conducted for the author's master's research.

The existing research investigates JEPA-based sequential recommendation and compares its performance with BERT4Rec. This course project instead focuses on network science: whether heterogeneous pairwise-comparison ranking can complement an established personalized recommender.

The BERT4Rec implementation and established preprocessing pipeline will be reused where appropriate to avoid unnecessary reimplementation and to preserve a consistent baseline.

## References

Newman, M. E. J. *Ranking with Multiple Types of Pairwise Comparisons*. Use the publication or preprint details corresponding to the version studied in the course.

Harper, F. Maxwell, and Joseph A. Konstan. 2015. “The MovieLens Datasets: History and Context.” *ACM Transactions on Interactive Intelligent Systems*, 5(4), Article 19. https://doi.org/10.1145/2827872.

MovieLens data is provided by the GroupLens Research Project at the University of Minnesota. Please follow the dataset's applicable usage terms.

## Author

Izaak King  
CSCI 6250 — Frontiers of Network Science  
Rensselaer Polytechnic Institute  
Fall 2026
