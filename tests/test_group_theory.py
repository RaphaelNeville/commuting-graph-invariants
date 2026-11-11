"""
Tests for the group-theoretic algorithms used in generate_groups_sage.py, run
against 13 groups (orders 4 to 27, p=2 and p=3) where the expected values are
known by hand or from standard references, plus one pair of groups of order
32 whose commuting graphs are isomorphic although their nilpotency classes
differ.

Run with:  pytest tests/test_group_theory.py -v

generate_groups_sage.py needs GAP/Sage, but these tests do not.
group_theory_lib.py reimplements the same definitions with explicit
multiplication tables, so nilpotency class, power subgroups, the
powerful-group check and the closed-form identities for the number of
conjugacy classes and the centre size can be checked anywhere.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from group_theory_lib import (
    Grp, cyclic, semidirect_zn_z2, semidirect_zn_zk, direct_product, quaternion8,
    heisenberg_group, to_dataset_json_entry
)


@pytest.fixture(scope="module")
def groups():
    return {
        "C4": cyclic(4),
        "C2xC2": semidirect_zn_z2(2, 1, "C2xC2"),
        "C8": cyclic(8),
        "D8": semidirect_zn_z2(4, 3, "D8"),
        "Q8": quaternion8(),
        "C2xC2xC2": direct_product(cyclic(2), direct_product(cyclic(2), cyclic(2))),
        "D16": semidirect_zn_z2(8, 7, "D16"),
        "M16": semidirect_zn_z2(8, 5, "M16"),
        "SD16": semidirect_zn_z2(8, 3, "SD16"),
    }


@pytest.fixture(scope="module")
def groups_p3():
    """Test groups for p=3, used to check the powerful-group code at an odd prime."""
    return {
        "C9": cyclic(9),
        "C3xC3": semidirect_zn_zk(3, 3, 1, "C3xC3"),
        "M27": semidirect_zn_zk(9, 3, 4, "M27"),       # powerful but not powerfully nilpotent
        "Heis3": heisenberg_group(3),                   # extraspecial, exponent 3, not powerful
    }


# --- known-by-hand ground truth ---
EXPECTED_CLASS = {
    "C4": 1, "C2xC2": 1, "C8": 1, "C2xC2xC2": 1,   # abelian, so class 1
    "D8": 2, "Q8": 2, "D16": 3, "M16": 2, "SD16": 3,
}
EXPECTED_KG = {  # number of conjugacy classes
    "C4": 4, "C2xC2": 4, "C8": 8, "C2xC2xC2": 8,
    "D8": 5, "Q8": 5, "D16": 7, "M16": 10, "SD16": 7,
}
EXPECTED_POWERFUL = {
    "C4": True, "C2xC2": True, "C8": True, "C2xC2xC2": True,
    "D8": False, "Q8": False, "D16": False, "SD16": False,
    "M16": True,  # the standard example of a non-abelian powerful 2-group
}
EXPECTED_POWERFUL_CLASS = {"M16": 2}


def test_group_axioms_hold(groups):
    # Grp.__init__ checks associativity on construction, so this test only
    # confirms that every fixture group was built.
    for name, G in groups.items():
        assert G.n > 0, name


@pytest.mark.parametrize("name", list(EXPECTED_CLASS.keys()))
def test_nilpotency_class(groups, name):
    assert groups[name].nilpotency_class() == EXPECTED_CLASS[name]


@pytest.mark.parametrize("name", list(EXPECTED_KG.keys()))
def test_number_of_conjugacy_classes(groups, name):
    assert len(groups[name].conjugacy_classes()) == EXPECTED_KG[name]


@pytest.mark.parametrize("name", list(EXPECTED_POWERFUL.keys()))
def test_is_powerful(groups, name):
    assert groups[name].is_powerful_p2() == EXPECTED_POWERFUL[name]


@pytest.mark.parametrize("name", list(EXPECTED_POWERFUL_CLASS.keys()))
def test_powerful_class(groups, name):
    assert groups[name].powerful_class() == EXPECTED_POWERFUL_CLASS[name]


# --- odd-prime (p=3) generalization ---
EXPECTED_CLASS_P3 = {"C9": 1, "C3xC3": 1, "M27": 2, "Heis3": 2}
EXPECTED_KG_P3 = {"C9": 9, "C3xC3": 9, "M27": 11, "Heis3": 11}
EXPECTED_POWERFUL_P3 = {"C9": True, "C3xC3": True, "M27": True, "Heis3": False}
# M27 (Z9 rtimes Z3) is powerful but not powerfully nilpotent. Its upper
# powerfully central series stalls at the centre (size 3) and never reaches
# all 27 elements. This can happen only for odd p (Traustason-Williams: every
# powerful 2-group is powerfully nilpotent), so its expected powerful_class
# is None.
EXPECTED_POWERFUL_CLASS_P3 = {"C9": 1, "C3xC3": 1, "M27": None, "Heis3": None}


@pytest.mark.parametrize("name", list(EXPECTED_CLASS_P3.keys()))
def test_nilpotency_class_p3(groups_p3, name):
    assert groups_p3[name].nilpotency_class() == EXPECTED_CLASS_P3[name]


@pytest.mark.parametrize("name", list(EXPECTED_KG_P3.keys()))
def test_number_of_conjugacy_classes_p3(groups_p3, name):
    assert len(groups_p3[name].conjugacy_classes()) == EXPECTED_KG_P3[name]


@pytest.mark.parametrize("name", list(EXPECTED_POWERFUL_P3.keys()))
def test_is_powerful_p3(groups_p3, name):
    assert groups_p3[name].is_powerful(p=3) == EXPECTED_POWERFUL_P3[name]


@pytest.mark.parametrize("name", list(EXPECTED_POWERFUL_CLASS_P3.keys()))
def test_powerful_class_p3(groups_p3, name):
    assert groups_p3[name].powerful_class(p=3) == EXPECTED_POWERFUL_CLASS_P3[name]


def test_burnside_and_center_identities_hold_for_p3_too(groups_p3):
    """Check Burnside's lemma and the centre-size identity on the p=3
    fixtures, as the p=2 tests below do."""
    for name, G in groups_p3.items():
        mean_centralizer = sum(len(G.centralizer(x)) for x in G.elements) / G.n
        assert mean_centralizer == pytest.approx(len(G.conjugacy_classes()))
        entry = to_dataset_json_entry(G, 1, p=3)
        assert sum(row[3] for row in entry["node_features"]) == len(G.center()), name


def test_burnside_identity_centralizer_mean_equals_num_conj_classes(groups):
    """k(G) == mean_x |C_G(x)| (Burnside's lemma)."""
    for name, G in groups.items():
        mean_centralizer = sum(len(G.centralizer(x)) for x in G.elements) / G.n
        assert mean_centralizer == pytest.approx(len(G.conjugacy_classes()))


def test_center_size_equals_sum_of_is_central_indicator(groups):
    """|Z(G)| equals the sum of the is_central node feature."""
    for name, G in groups.items():
        entry = to_dataset_json_entry(G, 1)
        indicator_sum = sum(row[3] for row in entry["node_features"])
        assert indicator_sum == len(G.center()), name


def test_centralizer_class_size_orbit_stabilizer(groups):
    """|C_G(x)| * |class(x)| == |G| for every x (orbit-stabilizer)."""
    for name, G in groups.items():
        for x in G.elements:
            assert len(G.centralizer(x)) * len(G.conjugacy_class(x)) == G.n


def test_power_subgroup_generator_construction_agrees_on_fixture(groups):
    """On the fixture groups, G^p (generated by the p-th powers of all
    elements) equals the subgroup generated by the p-th powers of a greedy
    generating set. The two need not agree for irregular p-groups.
    """
    for name, G in groups.items():
        correct = G.power_subgroup_correct(1)
        # greedy generating set
        S, cur = set(), {G.e}
        for x in G.elements:
            if x not in cur:
                S.add(x)
                cur = G.closure(S)
                if len(cur) == G.n:
                    break
        gen_based = G.power_subgroup_from_generators(1, S)
        assert correct == gen_based, name


def test_to_dataset_json_entry_schema(groups):
    """Check that to_dataset_json_entry() output has the fields dataset.py
    reads (order, group_id, num_nodes, edges, node_features with 5 per node,
    invariants)."""
    entries = [to_dataset_json_entry(G, i + 1) for i, G in enumerate(groups.values())]
    for e in entries:
        assert set(e.keys()) >= {"order", "group_id", "num_nodes", "edges", "node_features", "invariants"}
        assert len(e["node_features"]) == e["num_nodes"]
        assert all(len(row) == 5 for row in e["node_features"])
        required_inv = {"nilpotency_class", "derived_length", "center_size",
                         "commutator_size", "num_conj_classes"}
        assert required_inv <= set(e["invariants"].keys())
        max_idx = max((max(edge) for edge in e["edges"]), default=0)
        assert max_idx < e["num_nodes"]


# --- commuting-graph collisions between isoclinic groups ---

def _commuting_graph_structure(G):
    """(number of universal vertices, sorted [(component size, is clique)]
    after removing universal vertices). For graphs of the form
    'clique joined to disjoint cliques' this tuple is a complete isomorphism
    invariant, so equality of structures proves graph isomorphism."""
    n = G.n
    adj = {x: set() for x in G.elements}
    for x in G.elements:
        for y in G.centralizer(x):
            if y != x:
                adj[x].add(y)
    universal = {x for x in G.elements if len(adj[x]) == n - 1}
    rest = [x for x in G.elements if x not in universal]
    restset = set(rest)
    seen, comps = set(), []
    for s in rest:
        if s in seen:
            continue
        stack, comp = [s], set()
        while stack:
            v = stack.pop()
            if v in comp:
                continue
            comp.add(v)
            stack.extend(w for w in adj[v] & restset if w not in comp)
        seen |= comp
        comps.append(comp)
    comp_info = sorted(
        (len(c), all(len(adj[v] & c) == len(c) - 1 for v in c)) for c in comps
    )
    return len(universal), comp_info


def _five_targets(G):
    return (G.nilpotency_class(), G.derived_length(), len(G.center()),
            len(G.derived_subgroup()), len(G.conjugacy_classes()))


COLLISION_PAIRS = [
    # (constructor A, constructor B, expected structure, exponents differ?)
    (lambda: cyclic(4), lambda: semidirect_zn_z2(2, 1, "C2xC2"),
     (4, []), True),                                   # K4 = 4 universal vertices
    (lambda: semidirect_zn_z2(4, 3, "D8"), quaternion8,
     (2, [(2, True)] * 3), False),                     # K2 v 3.K2
    (lambda: heisenberg_group(3), lambda: semidirect_zn_zk(9, 3, 4, "M27"),
     (3, [(6, True)] * 4), True),                      # K3 v 4.K6
    (lambda: semidirect_zn_z2(8, 7, "D16"), lambda: semidirect_zn_z2(8, 3, "SD16"),
     (2, [(2, True)] * 4 + [(6, True)]), False),       # maximal-class pair, order 16
]


@pytest.mark.parametrize("makeA,makeB,expected,exp_differs", COLLISION_PAIRS)
def test_commuting_graph_collisions(makeA, makeB, expected, exp_differs):
    A, B = makeA(), makeB()
    sA, sB = _commuting_graph_structure(A), _commuting_graph_structure(B)
    # the same join-of-cliques structure gives isomorphic commuting graphs
    assert sA == expected and sB == expected
    # isoclinic groups of the same order agree on the five classical targets
    assert _five_targets(A) == _five_targets(B)
    # but the order multiset always differs across these pairs,
    # and the exponents differ if and only if exp_differs is True
    ordA = sorted(A.order_of(x) for x in A.elements)
    ordB = sorted(B.order_of(x) for x in B.elements)
    assert ordA != ordB
    assert (max(ordA) != max(ordB)) == exp_differs


# --- a non-isoclinic collision of order 32 that separates nilpotency class ---

def _wreath_c2xc2_by_c2():
    """(C2 x C2) wr C2 = C2^4 rtimes C2, where C2 swaps the two C2^2
    factors. Element (v, s), with v a 4-bit vector and s in {0, 1}, is
    encoded as 2*v + s, and (v, s)(w, t) = (v + swap^s(w), s + t)."""
    def swap(v):
        return ((v & 0b11) << 2) | (v >> 2)
    mul = [[0] * 32 for _ in range(32)]
    for x in range(32):
        v, s = x >> 1, x & 1
        for y in range(32):
            w, t = y >> 1, y & 1
            mul[x][y] = 2 * (v ^ (swap(w) if s else w)) + (s ^ t)
    return Grp(32, mul, "(C2xC2)wrC2")


def test_order32_collision_separating_nilpotency_class():
    """C2 x D16 (class 3) and (C2 x C2) wr C2 (class 2) both have commuting
    graph K4 v (K12 + 4.K4). Nilpotency class is an isoclinism invariant, so
    the two groups are not isoclinic, and nilpotency class is not determined
    by the commuting graph at order 32."""
    A = direct_product(cyclic(2), semidirect_zn_z2(8, 7, "D16"))
    B = _wreath_c2xc2_by_c2()
    assert (A.nilpotency_class(), B.nilpotency_class()) == (3, 2)
    assert A.derived_length() == B.derived_length() == 2
    assert len(A.center()) == len(B.center()) == 4
    assert len(A.derived_subgroup()) == len(B.derived_subgroup()) == 4
    assert len(A.conjugacy_classes()) == len(B.conjugacy_classes()) == 14
    # the same join-of-cliques structure gives isomorphic commuting graphs
    expected = (4, [(4, True)] * 4 + [(12, True)])
    assert _commuting_graph_structure(A) == _commuting_graph_structure(B) == expected
    # element orders separate the pair (R1): A has elements of order 8, B does not
    assert max(A.order_of(x) for x in A.elements) == 8
    assert max(B.order_of(x) for x in B.elements) == 4


# --- per-element pdepth values under the all-elements definition of G^(p^k) ---

def test_pdepth_exact_C8():
    """C8, p=2: generators have depth 0, squares that are not 4th powers
    depth 1, g^4 depth 2 and the identity depth 3 (log_2 of the exponent)."""
    from math import gcd
    entry = to_dataset_json_entry(cyclic(8), 1, p=2)
    for i, row in enumerate(entry["node_features"]):
        o = 8 // gcd(8, i) if i else 1
        expected = {8: 0, 4: 1, 2: 2, 1: 3}[o]
        assert row[4] == expected, (i, o, row[4], expected)


def test_pdepth_exact_D8_and_M27():
    d8 = to_dataset_json_entry(semidirect_zn_z2(4, 3, "D8"), 1, p=2)
    assert sorted(r[4] for r in d8["node_features"]) == [0] * 6 + [1, 2]
    m27 = to_dataset_json_entry(semidirect_zn_zk(9, 3, 4, "M27"), 1, p=3)
    assert sorted(r[4] for r in m27["node_features"]) == [0] * 24 + [1, 1, 2]


def test_pdepth_not_identically_zero_in_sample():
    """pdepth is non-zero for at least one node in sample_groups.json."""
    import json
    with open(os.path.join(os.path.dirname(__file__), "sample_groups.json")) as f:
        data = json.load(f)
    assert any(row[4] > 0 for e in data for row in e["node_features"])
