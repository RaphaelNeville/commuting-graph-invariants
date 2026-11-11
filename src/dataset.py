import json
import math
import random
import warnings
from collections import defaultdict

import torch
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import networkx as nx

try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceImage
    HAS_PH = True
except ImportError:
    HAS_PH = False


def _smallest_prime_factor(n):
    d = 2
    while d * d <= n:
        if n % d == 0:
            return d
        d += 1
    return n


class PGroupDataset(InMemoryDataset):
    """
    PyG dataset over the JSON produced by generate_groups_sage.py.

    Targets (5): nilpotency class, derived length, centre size, commutator
    size, number of conjugacy classes. With log_size_targets=True (default),
    centre size and commutator size are stored as log_p of the raw size,
    so they are integers for p-groups and comparable across primes.

    Persistent homology is computed on the noncentral commuting graph
    (central vertices removed). On the full commuting graph the identity is
    a universal vertex, so the Vietoris-Rips complex is a cone at every
    threshold >= 1 and H1 is identically zero, so a descriptor computed on
    the full graph would be trivial for every group. For abelian groups the
    noncentral graph is empty and the descriptor is zero by convention.
    One PersistenceImage transformer is fitted on the training diagrams only
    and applied to all splits.

    Split: orders with fewer than min_stratum_size groups go entirely into
    training. Remaining orders are split 80/10/10 within each order.
    Exact sizes are printed by report_split_by_order().
    """

    def __init__(self, path, use_ph=False, seed=42, stratify_by_order=True,
                 train_frac=0.8, val_frac=0.1, log_size_targets=True,
                 min_stratum_size=10, split_mode="stratified",
                 train_max_order=None, test_min_order=None, ph_strict=True):
        """split_mode="order_transfer" (with train_max_order, and optionally
        test_min_order) computes the order-transfer split inside the
        constructor, before the PersistenceImage is fitted. With use_ph=True
        the transformer is then fitted on the small-order training groups
        only. Use this for PH-enabled order-transfer experiments instead of
        the order_transfer_split() method, which runs after PH fitting."""
        super().__init__()

        if use_ph and not HAS_PH:
            raise ImportError(
                "use_ph=True requires giotto-tda. Install the optional PH "
                "dependencies or pass use_ph=False.")

        with open(path, "r") as f:
            raw = json.load(f)

        self.log_size_targets = log_size_targets
        data_list, self.extra = [], []
        dist_matrices = []       # per item: ndarray or None (for PH)

        for item in raw:
            edge_index = torch.tensor(item["edges"], dtype=torch.long).t().contiguous()
            x = torch.tensor(item["node_features"], dtype=torch.float)
            prime = item.get("prime") or _smallest_prime_factor(item["order"])

            inv = item["invariants"]
            centre, comm = inv["center_size"], inv["commutator_size"]
            if log_size_targets:
                centre = math.log(centre, prime)
                comm = math.log(comm, prime)
            y = torch.tensor([
                inv["nilpotency_class"], inv["derived_length"],
                centre, comm, inv["num_conj_classes"],
            ], dtype=torch.float).view(5)

            data_list.append(Data(
                x=x, edge_index=edge_index, y=y,
                ph_features=torch.zeros(100),
            ))
            self.extra.append({
                "order": item["order"], "prime": prime,
                "exponent": inv.get("exponent"),
                "is_powerful": inv.get("is_powerful"),
                "powerful_class": inv.get("powerful_class"),
                "powerful_coclass": inv.get("powerful_coclass"),
            })

            if use_ph and HAS_PH:
                # noncentral vertices, straight from the centrality feature
                keep = [i for i, row in enumerate(item["node_features"]) if row[3] == 0]
                if len(keep) >= 2:
                    g = nx.Graph()
                    g.add_nodes_from(keep)
                    keepset = set(keep)
                    g.add_edges_from((a, b) for a, b in item["edges"]
                                     if a in keepset and b in keepset)
                    dm = np.array(nx.floyd_warshall_numpy(g, nodelist=keep))
                    off = dm[~np.eye(len(keep), dtype=bool)]
                    finite = off[np.isfinite(off)]
                    fill = (finite.max() + 1.0) if finite.size else 1.0
                    dm[~np.isfinite(dm)] = fill
                    dist_matrices.append(dm)
                else:
                    dist_matrices.append(None)   # abelian: noncentral graph is empty
            else:
                dist_matrices.append(None)

        # ---- split (before PH so the PersistenceImage can be train-fitted) ----
        n = len(data_list)
        rng = random.Random(seed)
        if split_mode == "order_transfer":
            if train_max_order is None:
                raise ValueError("split_mode='order_transfer' requires train_max_order")
            orders_present = sorted({int(e["order"]) for e in self.extra})
            tmo = test_min_order
            if tmo is None:
                larger = [o for o in orders_present if o > train_max_order]
                if not larger:
                    raise ValueError(f"No groups with order > {train_max_order}.")
                tmo = min(larger)
            pool = [i for i, e in enumerate(self.extra) if int(e["order"]) <= train_max_order]
            self.test_idx = [i for i, e in enumerate(self.extra) if int(e["order"]) >= tmo]
            rng.shuffle(pool)
            n_val = max(1, int(0.2 * len(pool)))
            self.val_idx, self.train_idx = pool[:n_val], pool[n_val:]
        elif stratify_by_order:
            by_order = defaultdict(list)
            for idx, ex in enumerate(self.extra):
                by_order[int(ex["order"])].append(idx)
            self.train_idx, self.val_idx, self.test_idx = [], [], []
            for order, idxs in by_order.items():
                idxs = list(idxs)
                rng.shuffle(idxs)
                if len(idxs) < min_stratum_size:
                    self.train_idx += idxs
                    continue
                k = len(idxs)
                n_train = int(train_frac * k)
                n_val = int((train_frac + val_frac) * k)
                self.train_idx += idxs[:n_train]
                self.val_idx += idxs[n_train:n_val]
                self.test_idx += idxs[n_val:]
            for lst in (self.train_idx, self.val_idx, self.test_idx):
                rng.shuffle(lst)
        else:
            indices = list(range(n))
            rng.shuffle(indices)
            n_train = int(train_frac * n)
            n_val = int((train_frac + val_frac) * n)
            self.train_idx = indices[:n_train]
            self.val_idx = indices[n_train:n_val]
            self.test_idx = indices[n_val:]

        # ---- persistent homology, train-fitted ----
        if use_ph and HAS_PH:
            have = [i for i, dm in enumerate(dist_matrices) if dm is not None]
            if have:
                try:
                    vr = VietorisRipsPersistence(homology_dimensions=[1],
                                                 metric="precomputed")
                    diagrams = vr.fit_transform([dist_matrices[i] for i in have])
                    pos = {orig: row for row, orig in enumerate(have)}
                    n_nontrivial = sum(
                        1 for i in have
                        if np.any(diagrams[pos[i]][:, 1] > diagrams[pos[i]][:, 0])
                    )
                    print(f"[PH] noncentral H1 nontrivial for {n_nontrivial}/{len(have)} "
                          f"non-abelian graphs ({len(dist_matrices)-len(have)} empty by convention)")
                    train_rows = [pos[i] for i in self.train_idx if i in pos]
                    if not train_rows:
                        warnings.warn(
                            "No training-split graphs have PH diagrams. Fitting "
                            "the PersistenceImage on all diagrams, including "
                            "val and test. This leaks test information and is only suitable "
                            "for smoke tests.")
                    fit_rows = train_rows if train_rows else list(range(len(have)))
                    pi = PersistenceImage(n_bins=10)
                    pi.fit(diagrams[fit_rows])
                    images = pi.transform(diagrams)
                    for i in have:
                        vec = images[pos[i]].flatten()
                        if vec.shape[0] != 100:
                            raise ValueError(f"Expected PH dim 100, got {vec.shape[0]}")
                        data_list[i].ph_features = torch.tensor(vec, dtype=torch.float).view(-1)
                except Exception as e:
                    if ph_strict:
                        raise RuntimeError(
                            "PH computation failed. Pass ph_strict=False to "
                            "continue with zero PH features.") from e
                    warnings.warn(f"PH computation failed ({e}). Leaving ph_features at zero.")

        self.data, self.slices = self.collate(data_list)

    def report_split_by_order(self):
        counts = defaultdict(lambda: [0, 0, 0])
        for split_i, idxs in enumerate([self.train_idx, self.val_idx, self.test_idx]):
            for i in idxs:
                counts[int(self.extra[i]["order"])][split_i] += 1
        print(f"{'order':>6s} {'train':>6s} {'val':>6s} {'test':>6s}")
        for order in sorted(counts):
            t, v, te = counts[order]
            print(f"{order:6d} {t:6d} {v:6d} {te:6d}")

    def powerful_info(self, i):
        return self.extra[i]

    def order_transfer_split(self, train_max_order, test_min_order=None):
        """Train/val on order <= train_max_order, zero-shot test on larger.
        With use_ph=True the PersistenceImage has already been fitted on the
        constructor's training split, which by default includes groups that
        this method puts in the test set. Use split_mode="order_transfer" in
        the constructor instead."""
        orders_present = sorted({int(e["order"]) for e in self.extra})
        if test_min_order is None:
            larger = [o for o in orders_present if o > train_max_order]
            if not larger:
                raise ValueError(f"No groups with order > {train_max_order}.")
            test_min_order = min(larger)
        train_pool = [i for i, e in enumerate(self.extra) if int(e["order"]) <= train_max_order]
        test_idx = [i for i, e in enumerate(self.extra) if int(e["order"]) >= test_min_order]
        rng = random.Random(0)
        rng.shuffle(train_pool)
        n_val = max(1, int(0.2 * len(train_pool)))
        return train_pool[n_val:], train_pool[:n_val], test_idx
