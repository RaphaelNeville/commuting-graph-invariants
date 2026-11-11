"""
Non-GNN baselines on the same stratified split as notebooks/train.ipynb
(default seed 42, set with --seed):

  (a) constant training-set-mean predictor per target
  (b) linear regression on per-graph summary statistics
  (c) gradient boosting on the same summary statistics

Summary statistics per graph: num nodes, undirected edge count, degree
distribution moments (mean/std/min/max), centralizer-size histogram moments
(mean/std/min/max), counts per pdepth value, element-order histogram
(counts of elements of order p^j).

Targets come from PGroupDataset with its defaults, so centre and commutator
sizes are in log_p units, the same units as the notebook's reported MSEs.

Usage (from src/):
    python baselines.py ../data/commuting_pgroups_p2.json
"""

import argparse
import json

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from dataset import PGroupDataset

SEED = 42           # default, override with --seed
PDEPTH_BINS = 11    # pdepth values 0..10 (p=2, order <= 128 needs 0..7)
ORDER_BINS = 11     # element orders p^0..p^10


def moments(a):
    a = np.asarray(a, dtype=float)
    return [a.mean(), a.std(), a.min(), a.max()]


def graph_summary(item):
    nf = np.array(item["node_features"], dtype=float)
    n = item["num_nodes"]
    edges = item["edges"]
    m = len(edges) // 2  # edges stored bidirectionally

    deg = np.zeros(n)
    for a, b in edges:
        deg[a] += 0.5
        deg[b] += 0.5

    p = item.get("prime", 2)
    pdepth_counts = [0] * PDEPTH_BINS
    for d in nf[:, 4].astype(int):
        pdepth_counts[min(d, PDEPTH_BINS - 1)] += 1
    order_counts = [0] * ORDER_BINS
    for o in nf[:, 0].astype(int):
        j = int(round(np.log(o) / np.log(p))) if o > 1 else 0
        order_counts[min(j, ORDER_BINS - 1)] += 1

    return ([float(n), float(m)]
            + moments(deg)
            + moments(nf[:, 1])          # centralizer sizes
            + pdepth_counts
            + order_counts)


def main(path, seed=SEED):
    with open(path) as f:
        raw = json.load(f)

    dataset = PGroupDataset(path, use_ph=False, seed=seed, stratify_by_order=True)
    assert len(raw) == len(dataset), "raw JSON and dataset out of sync"

    X = np.array([graph_summary(item) for item in raw])
    Y = torch.stack([dataset[i].y for i in range(len(dataset))]).view(-1, 5).numpy()

    tr, te = dataset.train_idx, dataset.test_idx
    X_tr, X_te, Y_tr, Y_te = X[tr], X[te], Y[tr], Y[te]
    print(f"Split (seed {seed}, stratified): train {len(tr)}, test {len(te)}, "
          f"{X.shape[1]} summary features")

    names = ["Nilpotency class", "Derived length", "Centre size (log2 units)",
             "Commutator size (log2 units)", "Conjugacy classes"]

    results = {}

    # (a) constant training-set mean
    const_pred = Y_tr.mean(axis=0)
    results["constant_mean"] = ((Y_te - const_pred) ** 2).mean(axis=0)

    # (b) linear regression per target
    lin_mses = []
    for j in range(5):
        reg = LinearRegression().fit(X_tr, Y_tr[:, j])
        lin_mses.append(((reg.predict(X_te) - Y_te[:, j]) ** 2).mean())
    results["linear_regression"] = np.array(lin_mses)

    # (c) gradient boosting per target
    gb_mses = []
    for j in range(5):
        reg = GradientBoostingRegressor(random_state=seed).fit(X_tr, Y_tr[:, j])
        gb_mses.append(((reg.predict(X_te) - Y_te[:, j]) ** 2).mean())
    results["gradient_boosting"] = np.array(gb_mses)

    print("\n=== BASELINE TEST MSES (same units as train.ipynb) ===")
    header = f"{'target':30s}" + "".join(f"{k:>20s}" for k in results)
    print(header)
    for i, name in enumerate(names):
        row = "".join(f"{results[k][i]:20.3f}" for k in results)
        print(f"{name:30s}{row}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?", default="../data/commuting_pgroups_p2.json")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(args.data, seed=args.seed)
