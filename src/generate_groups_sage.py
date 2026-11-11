import json
import os
from sage.all import *
from sage.libs.gap.libgap import libgap

libgap.LoadPackage("smallgrp")

# ============================================================================
# CONFIG
# ============================================================================
# Groups of order P^n for 1 <= n <= MAX_POWER_N are generated, for the prime
# P below. Two ready-made configs:
#   P=2, MAX_POWER_N=7 gives 2668 groups of orders 2 to 128 (default)
#   P=3, MAX_POWER_N=6 gives 594 groups (1+2+5+15+67+504) of orders 3 to 729
P = 2
MAX_POWER_N = 7

# Computing is_powerful / powerful_class / powerful_coclass roughly doubles
# generation time for the largest orders (each is an extra O(|G|^2)-ish
# pass). Set to False for just the 6 classical invariants.
COMPUTE_POWERFUL_INVARIANTS = True

os.makedirs("../data", exist_ok=True)
OUTFILE = f"../data/commuting_pgroups_p{P}.json"  # run this script from inside src/


def to_py(obj):
    """Recursively convert Sage/GAP types to JSON-safe Python types."""
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    try:
        if hasattr(obj, "pyobject"):
            return obj.pyobject()
    except Exception:
        pass
    if hasattr(obj, "__int__"):
        try:
            return int(obj)
        except Exception:
            pass
    if isinstance(obj, (list, tuple)):
        return [to_py(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    return str(obj)


def nilpotency_class_libgap(G):
    """
    Nilpotency class via the lower central series:
      gamma_1 = G
      gamma_{i+1} = [gamma_i, G]
    Nontrivial abelian groups have class 1. Works for any p-group.
    A standalone Python version is in tests/group_theory_lib.py, with tests
    in tests/test_group_theory.py.
    """
    gamma = G
    c = 1
    while int(libgap.Size(gamma)) != 1:
        gamma_next = libgap.CommutatorSubgroup(gamma, G)
        if int(libgap.Size(gamma_next)) == int(libgap.Size(gamma)):
            break
        gamma = gamma_next
        if int(libgap.Size(gamma)) != 1:
            c += 1
    return int(c)


def power_subgroup_all_elements(G, ambient_elements, k, p=P):
    """
    G^(p^k) := < g^(p^k) : g in G >, computed from every element of G.
    ambient_elements should be G.Elements(), passed in to avoid recomputing it.
    """
    powers = [x ** (p ** k) for x in ambient_elements]
    return libgap.Subgroup(G, powers)


def power_depths(G, elements, p=P):
    """pdepth(x) = largest k >= 0 such that x in G^(p^k).

    Implementation: walk the descending tower G = G^(p^0) >= G^(p^1) >= ...
    and overwrite each surviving element's depth at every level, so the
    final value is the last level at which the element still appears.

    Convention for the identity: 1 lies in every power subgroup, so its
    depth is the terminal level e at which G^(p^e) = {1}, i.e.
    e = log_p(exponent of G).
    """
    pdepth_dict = {}
    k = 0
    current_elements = set(elements)  # k=0: G^(p^0) = G itself
    while True:
        for x in current_elements:
            pdepth_dict[x] = k
        if len(current_elements) == 1:
            break
        k += 1
        Gk = power_subgroup_all_elements(G, elements, k, p=p)
        new_elements = set(Gk.Elements())
        if new_elements == current_elements:
            # stabilized without reaching the identity, which cannot happen
            # for a finite p-group (G^(p^e) = 1 for p^e the exponent)
            break
        current_elements = new_elements
    return pdepth_dict


def is_powerful(G, elements, comm_subgroup, p=P):
    """
    G is powerful if [G,G] <= G^p for p odd, or [G,G] <= G^4 for p=2
    (Lubotzky-Mann; Dixon-du Sautoy-Mann-Segal, "Analytic pro-p groups"
    Ch.2; Traustason-Williams, "Powerfully nilpotent groups").
    """
    k = 1 if p != 2 else 2
    Gk = power_subgroup_all_elements(G, elements, k, p=p)
    Gk_elements = set(Gk.Elements())
    return all(x in Gk_elements for x in comm_subgroup.Elements())


def powerful_class(G, elements, powerful, p=P):
    """
    Upper powerfully central series (Traustason-Williams 2019, Sec.1):
        Zhat_0(G) = {1}
        Zhat_n(G) = { a in G : [a,x] in Zhat_{n-1}(G)^p  for all x in G }
    The series uses p-th powers for every p, as in the definition of a
    powerfully central chain, whereas is_powerful uses G^4 when p=2.
    Powerful class = smallest n with Zhat_n(G) = G.

    For p=2, every powerful group is powerfully nilpotent
    (Traustason-Williams, remark after the def. of strongly powerful), so
    this always returns a class when powerful is True. This does not
    generalize to odd p. The order-27 group Z9 rtimes Z3 (exponent 9) is
    powerful but not powerfully nilpotent: its upper series stalls at the
    centre and never reaches all of G.

    Returns None if G is not powerful, or if it is powerful but not
    powerfully nilpotent (possible only for odd p).
    """
    if not powerful:
        return None
    n_elems = len(elements)
    zhat_prev = {G.Identity()}
    for step in range(1, 30):
        zhat_prev_p = libgap.Subgroup(G, [z ** p for z in zhat_prev])
        zhat_prev_p_elements = set(zhat_prev_p.Elements())
        zhat_next = set()
        for a in elements:
            if all(libgap.Comm(a, x) in zhat_prev_p_elements for x in elements):
                zhat_next.add(a)
        if len(zhat_next) == n_elems:
            return step
        if zhat_next == zhat_prev:
            return None  # stalled below G: powerful but not powerfully nilpotent (possible for odd p)
        zhat_prev = zhat_next
    return None


def number_small_groups(n):
    return int(libgap.NumberSmallGroups(n))


print(f"Starting generation of {P}-groups with node features...\n")

dataset = []

for n in range(1, MAX_POWER_N + 1):
    N = P ** n
    print(f"\n=== Generating groups of order {N} ===")

    num_groups = number_small_groups(N)
    print(f" -> Found {num_groups} groups")

    for i in range(1, num_groups + 1):

        G = libgap.SmallGroup(N, i)

        elements = list(G.Elements())
        index_map = {g: idx for idx, g in enumerate(elements)}

        Z = G.Center()
        conj_classes = G.ConjugacyClasses()

        class_size = {}
        for C in conj_classes:
            size = int(C.Size())
            for x in C.Elements():
                class_size[x] = size

        comm = libgap.DerivedSubgroup(G)
        comm_size = int(libgap.Size(comm))

        ds = G.DerivedSeries()
        derived_length = len(ds) - 1

        exponent = int(libgap.Exponent(G))

        invariants = {
            "nilpotency_class": nilpotency_class_libgap(G),
            "derived_length":   int(derived_length),
            "center_size":      int(libgap.Size(Z)),
            "commutator_size":  comm_size,
            "num_conj_classes": int(len(conj_classes)),
            "exponent":         exponent,
        }

        if COMPUTE_POWERFUL_INVARIANTS:
            powerful = is_powerful(G, elements, comm, p=P)
            pclass = powerful_class(G, elements, powerful, p=P)
            invariants["is_powerful"] = bool(powerful)
            invariants["powerful_class"] = pclass
            invariants["powerful_coclass"] = (int(n) - pclass) if pclass is not None else None

        # Power depth of each element
        pdepth_dict = power_depths(G, elements, p=P)

        node_features = []
        Z_elements = set(Z.Elements())
        for x in elements:
            pdepth = pdepth_dict.get(x, 0)
            node_features.append([
                int(x.Order()),
                int(G.Centralizer(x).Size()),
                int(class_size[x]),
                1 if x in Z_elements else 0,
                pdepth,
            ])

        edges = []
        for idx_x, x in enumerate(elements):
            Cx = G.Centralizer(x).Elements()
            for y in Cx:
                if y != x:
                    idx_y = index_map[y]
                    if idx_x < idx_y:
                        edges.append([idx_x, idx_y])

        bidir_edges = edges + [[b, a] for a, b in edges]

        dataset.append({
            "order": int(N),
            "prime": int(P),
            "group_id": int(i),
            "num_nodes": len(elements),
            "edges": bidir_edges,
            "node_features": node_features,
            "invariants": invariants,
        })

with open(OUTFILE, "w") as f:
    json.dump(to_py(dataset), f, indent=2)

print(f"\nDataset saved to {OUTFILE}")
print("Total groups processed:", len(dataset))
print(f"(orders {P}^1 .. {P}^{MAX_POWER_N})")
