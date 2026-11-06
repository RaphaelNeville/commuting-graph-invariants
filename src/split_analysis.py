"""
Split-level analysis against the census's representation classes:

  (a) per seed: fraction of test groups whose R0 / R1 / R2 isomorphism
      class also contains at least one training group ("seen-representation"
      overlap).
  (b) per seed: split-specific Bayes floors, i.e. the minimum test MSE any
      predictor factoring through R can achieve on that test subset
      (within-class variance of the target over test members of each class,
      averaged over the test set), in the same units as the census's
      dataset-global floors (log_p for the two size targets).

Requires results/representation_classes.json (collision_census.py
--export_classes). Usage (from src/):
    python split_analysis.py ../data/commuting_pgroups_p2.json \
        ../results/representation_classes.json
"""

import json
import math
import sys
from collections import defaultdict

from dataset import PGroupDataset

TARGETS = ["nilpotency_class", "derived_length", "center_size",
           "commutator_size", "num_conj_classes", "exponent"]
SEEDS = [42, 43, 44, 45, 46]


def target_value(item, key):
    v = item["invariants"][key]
    if key in ("center_size", "commutator_size") and v > 0:
        return math.log(v, item.get("prime") or 2)
    return float(v)


def main(data_path, classes_path):
    raw = json.load(open(data_path))
    classes = json.load(open(classes_path))
    assert len(raw) == len(classes), "dataset and class export differ in length"
    for i, (item, row) in enumerate(zip(raw, classes)):
        assert (row["index"], row["order"], row["group_id"]) == \
               (i, item["order"], item["group_id"]), (
            f"class-export row {i} does not match the dataset (stale export?)")
    cls = {lvl: [row[lvl] for row in classes] for lvl in ("r0", "r1", "r2")}

    for seed in SEEDS:
        ds = PGroupDataset(data_path, use_ph=False, seed=seed,
                           stratify_by_order=True)
        tr, te = set(ds.train_idx), list(ds.test_idx)
        print(f"\n=== seed {seed}: train {len(tr)}, test {len(te)} ===")

        for lvl in ("r0", "r1", "r2"):
            train_classes = {cls[lvl][i] for i in tr}
            seen = sum(1 for i in te if cls[lvl][i] in train_classes)
            print(f"  {lvl.upper()} overlap: {seen}/{len(te)} test groups "
                  f"({seen/len(te):.4f}) have a {lvl.upper()}-isomorphic "
                  f"training partner")

        print(f"  split-specific test floors "
              f"({'/'.join(t for t in TARGETS)}):")
        for lvl in ("r0", "r1", "r2"):
            by_class = defaultdict(list)
            for i in te:
                by_class[cls[lvl][i]].append(i)
            floors = []
            for key in TARGETS:
                sq = 0.0
                for members in by_class.values():
                    vals = [target_value(raw[i], key) for i in members]
                    m = sum(vals) / len(vals)
                    sq += sum((v - m) ** 2 for v in vals)
                floors.append(sq / len(te))
            # is_powerful: 0-1 min-error classification floor on the test set
            err = 0
            for members in by_class.values():
                flags = [bool(raw[i]["invariants"]["is_powerful"]) for i in members]
                err += min(flags.count(True), flags.count(False))
            print(f"    {lvl.upper()}: "
                  + "  ".join(f"{k}={f:.5f}" for k, f in zip(TARGETS, floors))
                  + f"  is_powerful={err/len(te):.5f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../data/commuting_pgroups_p2.json",
         sys.argv[2] if len(sys.argv) > 2 else "../results/representation_classes.json")
