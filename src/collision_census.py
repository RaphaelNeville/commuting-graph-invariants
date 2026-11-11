"""
Commuting-graph collision census, at three representation levels.

Finds all collision classes (sets of two or more groups with isomorphic
commuting graphs) and reports, per class, which invariants vary. Collision
classes are computed at three levels:

  R0: unlabelled graph isomorphism
  R1: graph isomorphism preserving element-order node labels
  R2: graph isomorphism preserving (element order, power depth) labels

A predictor whose input is representation R cannot distinguish groups inside
one R-collision class, so within-class variance of a target is a floor on
the MSE achievable at that level (the Bayes-floor proposition in the
paper). The floors are printed per target and per level. R2 classes refine
R1 classes, which refine R0 classes, so the floors cannot increase with
level. The trained model reads R2 input or richer, so only the R2 floors
bound its error.

Known validation cases from the literature:
  * order 128: Schur covers of SmallGroup(64,182) vs SmallGroup(64,182)xC2
    collide with commutator sizes differing by a factor of 2
    (Arvind-Cameron-Ma-Maslova, J. Algebra 703 (2026); arXiv:2305.07301).
  * order 64: non-isoclinic collisions found by E. O'Brien (reported ids
    (64,18)/(64,215) in the same paper), refuting the class-2 conjecture of
    earlier arXiv versions of that paper. When order-64 groups are present,
    the census reports whether this pair collides.

Method: bucket by (order, WL hash) as a fast necessary filter, then group
each bucket by BLISS canonical form (python-igraph): two graphs are
isomorphic iff their canonical forms are identical, so R0 needs only one
canonicalization per graph. This avoids pairwise VF2-style isomorphism
search, which can take exponential time on dense, highly symmetric graphs
like these. The labelled refinements R1 and R2 run inside confirmed
R0 classes only: prebucket by label-seeded WL hash, then confirm pairwise
with igraph's coloured BLISS search (isomorphic_bliss).
Requires networkx + python-igraph (no torch, no Sage).

Usage:
    python collision_census.py ../data/commuting_pgroups_p2.json
    python collision_census.py ../tests/sample_groups.json     # quick demo
"""

import argparse
import json
import math
from collections import Counter, defaultdict

import igraph as ig
import networkx as nx

INVARIANT_KEYS = [
    "nilpotency_class", "derived_length", "center_size",
    "commutator_size", "num_conj_classes", "exponent",
    "is_powerful", "powerful_class",
]
FIVE_TARGETS = INVARIANT_KEYS[:5]
SCHUR_COVER_IDS_128 = {789, 790, 791, 815, 816, 817}
REPORTED_OBRIEN_PAIR_64 = {18, 215}

# Node-label attribute names for the labelled refinement levels.
# R1 uses element order, R2 the (order, pdepth) tuple.
MATCH_R1 = "order"
MATCH_R2 = "r2"


def build_graph(item):
    g = nx.Graph()
    for i, row in enumerate(item["node_features"]):
        g.add_node(i, order=row[0], pdepth=row[4], r2=(row[0], row[4]))
    g.add_edges_from(map(tuple, item["edges"]))
    return g


def group_label(item):
    return f"(order {item['order']}, id {item['group_id']})"


def order_multiset(item):
    return tuple(sorted(row[0] for row in item["node_features"]))


_color_ids = {}   # shared label-to-int map, so colours match across graphs
_ig_cache = {}    # igraph copies by id(g), colour vectors by (id(g), label)


def _ig_colored(g, node_label):
    """igraph copy of g plus integer colour vector for node_label, cached.
    Nodes must be 0..n-1 ints, as build_graph produces."""
    key = id(g)
    if key not in _ig_cache:
        _ig_cache[key] = ig.Graph(n=g.number_of_nodes(), edges=list(g.edges()))
    ckey = (key, node_label)
    if ckey not in _ig_cache:
        cols = []
        for i in range(g.number_of_nodes()):
            v = g.nodes[i][node_label]
            if v not in _color_ids:
                _color_ids[v] = len(_color_ids)
            cols.append(_color_ids[v])
        _ig_cache[ckey] = cols
    return _ig_cache[key], _ig_cache[ckey]


def _canonical_key(g):
    """BLISS canonical form of an uncoloured graph: a hashable key equal for
    two graphs iff they are isomorphic. Ignores node labels; the labelled
    levels R1 and R2 are confirmed with pairwise coloured BLISS search in
    partition_isomorphic."""
    igg = ig.Graph(n=g.number_of_nodes(), edges=list(g.edges()))
    perm = igg.canonical_permutation()
    gc = igg.permute_vertices(perm)
    return tuple(sorted(tuple(sorted(e)) for e in gc.get_edgelist()))


def partition_isomorphic(idxs, graphs, node_label=None):
    """Partition a list of indices into exact isomorphism classes.

    node_label=None (R0): group by BLISS canonical form, with one
    canonicalization per graph and no pairwise search.

    node_label="order" (R1) / "r2" (R2): called only on already-confirmed
    R0 classes. Prebucket by label-seeded WL hash (a necessary condition
    for labelled isomorphism), then confirm survivors pairwise with
    igraph's coloured BLISS search (isomorphic_bliss with color1/color2)."""
    if node_label is None:
        classes = {}
        for i in idxs:
            classes.setdefault(_canonical_key(graphs[i]), []).append(i)
        return list(classes.values())

    prebuckets = {}
    for i in idxs:
        h = nx.weisfeiler_lehman_graph_hash(graphs[i], node_attr=node_label,
                                            iterations=3)
        prebuckets.setdefault(h, []).append(i)
    classes = []
    for sub in prebuckets.values():
        remaining = list(sub)
        while remaining:
            rep = remaining.pop(0)
            cls, keep = [rep], []
            g_rep, c_rep = _ig_colored(graphs[rep], node_label)
            for other in remaining:
                g_oth, c_oth = _ig_colored(graphs[other], node_label)
                if g_rep.isomorphic_bliss(g_oth, color1=c_rep, color2=c_oth):
                    cls.append(other)
                else:
                    keep.append(other)
            remaining = keep
            classes.append(cls)
    return classes


def floors(collision_classes, raw, n_total):
    """Per-target irreducible MSE floors over the given collision classes,
    in the units used for training: centre and commutator size are
    log_p-transformed (matching dataset.py's log_size_targets=True),
    everything else raw. The powerful_class floor is divided by the number
    of groups where powerful_class is defined rather than by n_total.
    is_powerful gets a 0-1 classification floor."""
    out = {}
    n_pc_defined = sum(
        1 for item in raw if item["invariants"].get("powerful_class") is not None)
    for key in [k for k in INVARIANT_KEYS if k != "is_powerful"]:
        sq = 0.0
        for _order, cls in collision_classes:
            vals = []
            for i in cls:
                v = raw[i]["invariants"].get(key)
                if not isinstance(v, (int, float)):
                    continue
                if key in ("center_size", "commutator_size") and v > 0:
                    v = math.log(v, raw[i].get("prime") or 2)
                vals.append(v)
            if len(vals) >= 2:
                m = sum(vals) / len(vals)
                sq += sum((v - m) ** 2 for v in vals)
        denom = n_pc_defined if key == "powerful_class" else n_total
        out[key] = (sq / denom) if denom else 0.0
    err = 0
    for _order, cls in collision_classes:
        flags = [raw[i]["invariants"].get("is_powerful") for i in cls]
        flags = [f for f in flags if f is not None]
        if len(set(flags)) > 1:
            err += min(flags.count(True), flags.count(False))
    out["is_powerful"] = err / n_total
    return out


def export_representation_classes(raw, graphs, buckets, out_path):
    """Assign every group an R0, R1 and R2 class ID (sequential ints, with
    groups sharing an ID iff their graphs are isomorphic at that level) and
    write them to out_path as a JSON list aligned with the dataset order."""
    r0_id = [None] * len(raw)
    r1_id = [None] * len(raw)
    r2_id = [None] * len(raw)
    counters = [0, 0, 0]
    for (_order, _h), idxs in sorted(buckets.items()):
        r0_cls = ([[idxs[0]]] if len(idxs) == 1
                  else partition_isomorphic(idxs, graphs))
        for cls in r0_cls:
            for i in cls:
                r0_id[i] = counters[0]
            counters[0] += 1
            for level, label, ids in ((1, MATCH_R1, r1_id), (2, MATCH_R2, r2_id)):
                subs = ([cls] if len(cls) == 1
                        else partition_isomorphic(cls, graphs, label))
                for sub in subs:
                    for i in sub:
                        ids[i] = counters[level]
                    counters[level] += 1
    out = [{"index": i, "order": item["order"], "group_id": item["group_id"],
            "r0": r0_id[i], "r1": r1_id[i], "r2": r2_id[i]}
           for i, item in enumerate(raw)]
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"[export] wrote {len(out)} rows ({counters[0]} R0 / {counters[1]} R1 / "
          f"{counters[2]} R2 classes) to {out_path}")


def main(path, export_classes=None):
    with open(path) as f:
        raw = json.load(f)
    graphs = [build_graph(item) for item in raw]
    n_total = len(raw)

    print(f"Loaded {n_total} groups from {path}")
    print("Computing WL hashes...")
    buckets = defaultdict(list)
    for idx, (item, g) in enumerate(zip(raw, graphs)):
        h = nx.weisfeiler_lehman_graph_hash(g, iterations=3)
        buckets[(item["order"], h)].append(idx)
    if export_classes:
        export_representation_classes(raw, graphs, buckets, export_classes)

    candidates = {k: v for k, v in buckets.items() if len(v) > 1}
    print(f"{len(candidates)} candidate buckets; confirming by canonical form...\n")

    r0_classes = []
    for bi, ((order, _h), idxs) in enumerate(sorted(candidates.items()), 1):
        print(f"[progress] R0 bucket {bi}/{len(candidates)} "
              f"(order {order}, {len(idxs)} graphs)...", flush=True)
        for cls in partition_isomorphic(idxs, graphs):
            if len(cls) > 1:
                r0_classes.append((order, cls))

    if not r0_classes:
        print("No collisions found at any level.")
        return

    # refine R0 classes by labelled isomorphism
    r1_classes, r2_classes = [], []
    for ci, (order, cls) in enumerate(r0_classes, 1):
        print(f"[progress] labelled refinement {ci}/{len(r0_classes)} "
              f"(order {order}, class size {len(cls)})...", flush=True)
        for sub in partition_isomorphic(cls, graphs, MATCH_R1):
            if len(sub) > 1:
                r1_classes.append((order, sub))
        for sub in partition_isomorphic(cls, graphs, MATCH_R2):
            if len(sub) > 1:
                r2_classes.append((order, sub))

    print(f"=== {len(r0_classes)} confirmed R0 collision classes "
          f"({len(r1_classes)} survive R1 labels, {len(r2_classes)} survive R2) ===\n")
    any_five_target_split = False
    for order, cls in r0_classes:
        members = [raw[i] for i in cls]
        print(f"Order {order}: {', '.join(group_label(m) for m in members)}")
        for key in INVARIANT_KEYS:
            vals = [m["invariants"].get(key) for m in members]
            if key == "powerful_class" and any(v is None for v in vals) and any(v is not None for v in vals):
                print(f"    {key} is defined for only some members: {vals}")
                continue
            if len(set(map(str, vals))) > 1:
                marker = "  (classical target)" if key in FIVE_TARGETS else ""
                print(f"    {key} varies: {vals}{marker}")
                if key in FIVE_TARGETS:
                    any_five_target_split = True
        oms = [order_multiset(m) for m in members]
        if len(set(oms)) > 1:
            print(f"    element-order multiset varies: {[dict(Counter(o)) for o in oms]}")
        print()

    print("=== SUMMARY ===")
    print(f"Groups in R0 collisions: {sum(len(c) for _o, c in r0_classes)} / {n_total}")
    fl = {"R0": floors(r0_classes, raw, n_total),
          "R1": floors(r1_classes, raw, n_total),
          "R2": floors(r2_classes, raw, n_total)}
    LABELS = {"center_size": "center_size (log_p)",
              "commutator_size": "commutator_size (log_p)",
              "powerful_class": "powerful_class (cond.)"}
    print("\nIrreducible MSE floors by representation level, in training units.")
    print("The two size targets are log_p-transformed, matching log_size_targets=True.")
    print("R0: unlabelled graph")
    print("R1: graph with element-order labels")
    print("R2: graph with (element order, power depth) labels")
    print("Only the R2 column bounds the trained model, whose input is at least as rich as R2.")
    print("is_powerful: 0-1 classification floor")
    print("powerful_class: divided by the number of groups where it is defined")
    print("*: positive R0 floor")
    print(f"  {'target':26s} {'R0':>10s} {'R1':>10s} {'R2':>10s}")
    for key in INVARIANT_KEYS:
        row = "".join(f"{fl[lvl][key]:10.4f}" for lvl in ("R0", "R1", "R2"))
        star = "  *" if fl["R0"][key] > 0 else ""
        print(f"  {LABELS.get(key, key):26s} {row}{star}")

    # --- expected validation: the order-128 Arvind-Cameron-Ma-Maslova pair ---
    if any(item["order"] == 128 for item in raw):
        hit = False
        for order, cls in r0_classes:
            if order != 128:
                continue
            if not any(raw[i]["group_id"] in SCHUR_COVER_IDS_128 for i in cls):
                continue
            comms = sorted(raw[i]["invariants"]["commutator_size"] for i in cls)
            if len(set(comms)) > 1 and comms[-1] == 2 * comms[0]:
                hit = True
        if hit:
            print("\nVALIDATION: the known order-128 collision (Schur covers of "
                  "SmallGroup(64,182) vs SmallGroup(64,182)xC2, commutator sizes "
                  "in ratio 2) was recovered.")
        else:
            print("\nWARNING: dataset contains order-128 groups but the known "
                  "order-128 collision (Schur-cover id present and commutator "
                  "sizes in ratio 2 within the class) was not recovered. "
                  "Investigate before trusting the census.")

    # --- informational: reported O'Brien order-64 pair ---
    if any(item["order"] == 64 for item in raw):
        confirmed = any(
            order == 64 and REPORTED_OBRIEN_PAIR_64 <= {raw[i]["group_id"] for i in cls}
            for order, cls in r0_classes
        )
        if confirmed:
            print("INFO: the reported O'Brien order-64 non-isoclinic collision "
                  "(64,18)/(64,215) was confirmed as a collision in this run.")
        else:
            print("INFO: the reported O'Brien order-64 pair (64,18)/(64,215) was "
                  "not found as a collision in this run. The ids come from a literature "
                  "report, so check them against J. Algebra 703 (2026).")

    if any_five_target_split:
        print("\nAt least one collision separates one of the five classical targets:")
        print("by the isoclinism theorem, that pair must be non-isoclinic.")
    else:
        print("\nNo collision separates any of the five classical targets in this")
        print("dataset, so their R0 Bayes floor here is zero.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?", default="../tests/sample_groups.json")
    parser.add_argument("--export_classes", default=None,
                        help="also write per-group R0/R1/R2 class IDs to this JSON path")
    args = parser.parse_args()
    main(args.data, export_classes=args.export_classes)
