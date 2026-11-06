"""
Fast smoke test for the PyTorch side (dataset.py and model.py) using the
13-group sample_groups.json instead of the full 2,668-group dataset.
Requires torch/torch_geometric (and optionally giotto-tda):

    pip install -r requirements.txt
    pytest tests/test_pipeline_smoke.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_groups.json")


def test_dataset_loads_without_ph():
    from dataset import PGroupDataset
    ds = PGroupDataset(SAMPLE_PATH, use_ph=False, stratify_by_order=True)
    assert len(ds) == 13
    assert ds[0].x.shape[1] == 5, "node features should be 5-dimensional"
    assert ds[0].y.shape[0] == 5, "y should have 5 classical-invariant targets"
    # train, val and test sizes sum to 13
    assert len(ds.train_idx) + len(ds.val_idx) + len(ds.test_idx) == 13


def test_order_transfer_split():
    from dataset import PGroupDataset
    ds = PGroupDataset(SAMPLE_PATH, use_ph=False, stratify_by_order=False)
    train_idx, val_idx, test_idx = ds.order_transfer_split(train_max_order=16)
    # everything <= order 16 goes to train/val, order 27 (never seen) goes to test
    assert len(train_idx) + len(val_idx) + len(test_idx) == 13
    for i in test_idx:
        assert int(ds.powerful_info(i)["order"]) > 16
    for i in train_idx + val_idx:
        assert int(ds.powerful_info(i)["order"]) <= 16


def test_model_forward_pass_shapes():
    from torch_geometric.loader import DataLoader
    from dataset import PGroupDataset
    from model import PGroupGNN

    ds = PGroupDataset(SAMPLE_PATH, use_ph=False, stratify_by_order=False)
    loader = DataLoader([ds[i] for i in range(len(ds))], batch_size=4)
    model = PGroupGNN(node_dim=ds[0].x.shape[1], ph_dim=0, out_dim=5)

    batch = next(iter(loader))
    out = model(batch.x, batch.edge_index, batch.batch, batch.ph_features)
    assert out.shape == (batch.num_graphs, 5), out.shape


def test_model_forward_pass_with_ph_if_available():
    from torch_geometric.loader import DataLoader
    from dataset import PGroupDataset, HAS_PH
    from model import PGroupGNN

    if not HAS_PH:
        pytest.skip("giotto-tda not installed in this environment")

    ds = PGroupDataset(SAMPLE_PATH, use_ph=True, stratify_by_order=False)
    loader = DataLoader([ds[i] for i in range(len(ds))], batch_size=4)
    model = PGroupGNN(node_dim=ds[0].x.shape[1], ph_dim=100, out_dim=5)

    batch = next(iter(loader))
    out = model(batch.x, batch.edge_index, batch.batch, batch.ph_features)
    assert out.shape == (batch.num_graphs, 5), out.shape
    assert batch.ph_features.shape[-1] == 100 or batch.ph_features.numel() % 100 == 0


def test_powerful_group_baseline_matches_hand_computed_values():
    """Check that M16 in the sample fixture (the order-16 powerful group
    with 10 conjugacy classes) has powerful class 2 and nilpotency class 2,
    matching the hand-computed values in test_group_theory.py."""
    import json
    with open(SAMPLE_PATH) as f:
        raw = json.load(f)
    m16 = next(e for e in raw if e["order"] == 16 and e["invariants"]["is_powerful"]
               and e["invariants"]["num_conj_classes"] == 10)
    assert m16["invariants"]["powerful_class"] == 2
    assert m16["invariants"]["nilpotency_class"] == 2


def test_order_transfer_constructor_split():
    """The order-transfer split is computed in the constructor
    (split_mode="order_transfer"), before any PersistenceImage fitting,
    and the test set is exactly the groups of order above 16 (here all
    of order 27)."""
    from dataset import PGroupDataset
    ds = PGroupDataset(SAMPLE_PATH, use_ph=False,
                       split_mode="order_transfer", train_max_order=16)
    assert len(ds.train_idx) + len(ds.val_idx) + len(ds.test_idx) == 13
    for i in ds.test_idx:
        assert int(ds.powerful_info(i)["order"]) > 16
    for i in ds.train_idx + ds.val_idx:
        assert int(ds.powerful_info(i)["order"]) <= 16
    assert {int(ds.powerful_info(i)["order"]) for i in ds.test_idx} == {27}
