# Learning Algebraic Invariants of Finite p-Groups from Commuting Graphs

The code in this repository trains a graph neural network to predict five
structural invariants of a finite 2-group (nilpotency class, derived length,
centre size, commutator size and number of conjugacy classes) from its
*commuting graph*. The vertices of this graph are the group elements, with
an edge for each commuting pair, and each vertex carries five algebraic
features. The accompanying papers use proofs and a census of groups with
isomorphic commuting graphs to find the lowest error (the Bayes floor) that
any model reading this representation can reach on each invariant.

The code reproduces the five-target pipeline of the papers, from generating
the groups to training the model, as well as the collision census, the
baselines and the split analysis. The mathematical test suite is in
`tests/`. Generated datasets, model checkpoints and raw execution logs are
not included. The papers also report experiments that the workflow below
does not include (feature ablations, order transfer, cross-prime
fine-tuning and the powerful-groups study). Tested with Python 3.10. The
direct dependencies, at the versions used in testing, are listed in
`requirements-lock.txt`.

A 4-page summary of the results is in [`PROJECT_REPORT.pdf`](PROJECT_REPORT.pdf),
and the full paper (24 pages) is [`paper.pdf`](paper.pdf).

## Results

The dataset consists of the 2,668 nontrivial 2-groups of orders 2–128. The
census works at three levels of representation. R0 is the bare commuting
graph, R1 adds element orders and R2 adds power depth. The GNN and the
gradient-boosting baseline both receive R2-level information (the boosting
summaries include element-order and power-depth counts), so the R2 floors,
not the R0 floors, are the ones that bound their errors.

| target | GNN, 5 seeds (MSE) | training-set mean | grad. boosting on summaries | census Bayes floor at R0 | census Bayes floor at R2 |
|---|---|---|---|---|---|
| nilpotency class | 1.73 ± 3.22 | 0.53 ± 0.09 | 0.034 ± 0.005 | 0.141 | 0.0005 |
| derived length | 0.020 ± 0.019 | 0.022 ± 0.016 | 0.000 | 0 | 0 |
| centre size (log₂) | 0.45 ± 0.32 | 0.90 ± 0.19 | 0.011 ± 0.004 | 0 | 0 |
| commutator size (log₂) | 1.05 ± 1.61 | 0.72 ± 0.12 | 0.014 ± 0.002 | 0.022 | 0.0047 |
| conjugacy classes | 188.5 ± 295.1 | 211.6 ± 70.4 | 0.000 | 0 | 0 |

- The exhaustive collision census (WL-hash bucketing followed by BLISS
  canonical forms) finds 74 classes of groups with isomorphic commuting
  graphs, covering 2,667 of the 2,668 groups, and computes the Bayes floor
  of each target at R0, R1 and R2. It recovers the known Arvind–Cameron
  order-128 collision as a validation case. For groups of order at most 128
  it also answers two questions that the theory leaves open, showing that
  nilpotency class is separated by collisions (its floor stays positive
  even with all node labels) and that no collision separates derived
  length. The first collision that separates nilpotency class has order 32:
  C₂ × D₁₆ (class 3) and (C₂ × C₂) ≀ C₂ (class 2) have isomorphic commuting
  graphs, which `tests/test_group_theory.py` checks independently of the
  census.
  Collisions of this kind also follow from published results of Kakkar and
  Rawat and of Pezzott and Nakaoka, although neither paper discusses
  nilpotency class.
- Centre size and the number of conjugacy classes are proved to be
  closed-form functions of the input features (the second via Burnside's
  lemma). These formulas are reported as baselines that need no training
  (MSE 0.000), and their zero error is not credited to the model.
- Gradient boosting on summary statistics of each graph beats the GNN on
  every target in each of the five seeds. Averaged over the five seeds, the
  GNN is also worse than predicting the training-set mean on nilpotency
  class and commutator size. This is mainly due to one failed seed (42):
  over the other four seeds it averages 0.29 and 0.34 on these targets,
  against 0.55 and 0.72 for the training mean. The node features enter in
  raw units and only one configuration was tried, so this is a result about
  this pipeline rather than about GNNs on commuting graphs in general.
- The split analysis, which applies the census to the train/test split,
  shows that every test group shares its unlabelled graph representation
  with a training group, and 41–48% share their fully labelled class. The
  pooled test errors therefore
  measure interpolation within representations largely seen in training.
  Zero-shot order transfer, with training on orders up to 64 and testing on
  order 128, is a stronger test of generalization, and performance degrades
  under it (see the full paper).

## What's in this repo

| file | what it is |
|---|---|
| `PROJECT_REPORT.pdf` | 4-page summary of the main results |
| `paper.pdf` | full paper: theory, census, experiments |
| `src/generate_groups_sage.py` | builds the dataset from GAP's SmallGroups (any prime p) |
| `src/dataset.py` | PyG dataset: graphs, stratified splits, persistent homology |
| `src/model.py` | the 3-layer GIN with optional PH branch |
| `notebooks/train.ipynb` | the training loop and closed-form baselines |
| `src/collision_census.py` | the census: collision classes, Bayes floors, class export |
| `src/baselines.py` | constant, linear and gradient-boosting baselines |
| `src/split_analysis.py` | the split analysis: representation overlap and split-specific floors |
| `tests/` | 59 pure-Python tests of the group theory (no Sage/torch needed) and 6 smoke tests of the torch/PyG pipeline |

## Quickstart

```bash
pip install -r requirements.txt          # flexible install, includes jupyter
# or the tested direct-dependency versions (add jupyter separately):
# pip install -r requirements-lock.txt && pip install jupyter

# 1. Verify the group theory (pure Python, no Sage/torch), then smoke-test
#    the torch/PyG data loading, splits and forward pass on the 13-group fixture:
pytest tests/test_group_theory.py -v
pytest tests/test_pipeline_smoke.py -v

# 2. Census demo on the 13-group fixture (needs only networkx and python-igraph):
(cd src && python collision_census.py ../tests/sample_groups.json)

# 3. Full pipeline (needs SageMath with GAP's smallgrp package):
(cd src && sage generate_groups_sage.py)                     # -> data/ (2,668 groups, ~4 min)
mkdir -p results
(cd src && python collision_census.py ../data/commuting_pgroups_p2.json \
    --export_classes ../results/representation_classes.json) # census + floors (~40 s)
jupyter notebook notebooks/train.ipynb                       # training (~5-20 min/seed, CPU)
(cd src && python baselines.py ../data/commuting_pgroups_p2.json --seed 42)
(cd src && python split_analysis.py ../data/commuting_pgroups_p2.json \
    ../results/representation_classes.json)                  # the split analysis
```

The five reported runs use seeds 42 to 46. To reproduce them, set `SEED` in
the first code cell of the notebook, and `--seed` for the baselines, to each
of these values in turn.

## Use of AI tools

AI tools were used for programming and for checking computations.

## Licence

The code is released under the MIT licence (see [`LICENSE`](LICENSE)). The
paper and the project report (`paper.pdf` and `PROJECT_REPORT.pdf`) are
© 2026 Raphael Neville and are not covered by the MIT licence.
