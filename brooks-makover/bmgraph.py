"""
bmgraph.py -- combinatorial ("graph stage") features of a Brooks-Makover surface.

Everything here needs only the ribbon graph (tau, sigma) and is cheap compared
with the FEM stage, so it can be run on 10^5 - 10^7 samples for screening.

Features
--------
* adjacency spectrum of the cubic (multi)graph on the 2n triangles: mu_2, mu_min,
  spectral gap 3 - mu_2, Ramanujan flag (max(|mu_2|, |mu_min|) <= 2 sqrt 2)
* non-backtracking (Hashimoto) spectrum on the 6n darts: rho_2 = second largest
  modulus, Ramanujan iff rho_2 <= sqrt 2
* short closed non-backtracking walks up to length W (exact enumeration):
  girth, simple-cycle counts c_1..c_W (c_1 = loops, c_2 = double edges),
  tangle proxy = closed NB walks that are not simple cycles (two cycles close together)
* cusp statistics: number of cusps, lengths k_c, fraction of the largest cusp,
  number of cusps of length 1 and 2
* length spectrum of the cusped surface S = H/Gamma, Gamma < PSL(2,Z):
  a closed geodesic <-> a primitive cyclically reduced closed non-backtracking walk
  in the dual cubic graph whose turn word (left = sigma, right = sigma^{-1}) is not
  constant; with L = [[1,1],[0,1]], R = [[1,0],[1,1]] the geodesic length is
  2 arccosh(tr(word)/2).  Constant words go once around a face = a cusp (trace 2).
  Words up to length W give the COMPLETE length spectrum below
  ell_cut = 2 arccosh((W+2)/2)  (a word of length w has trace >= w+2).
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from bmsurf import BMSurface


# ----------------------------------------------------------------------------
# matrices
# ----------------------------------------------------------------------------
def adjacency(surf: BMSurface) -> sp.csr_matrix:
    N = 2 * surf.n
    rows = np.arange(6 * surf.n) // 3
    cols = surf.tau // 3
    return sp.coo_matrix((np.ones(6 * surf.n), (rows, cols)), shape=(N, N)).tocsr()


def nonbacktracking(surf: BMSurface) -> sp.csr_matrix:
    """B[d, d'] = 1 iff d' leaves the head vertex of d and d' != tau(d)."""
    D = 6 * surf.n
    head = surf.tau // 3
    rows, cols = [], []
    for d in range(D):
        v = head[d]
        for s in range(3):
            dp = 3 * v + s
            if dp != surf.tau[d]:
                rows.append(d); cols.append(dp)
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(D, D)).tocsr()


def adjacency_spectrum(surf: BMSurface, dense_max=4096):
    A = adjacency(surf)
    N = A.shape[0]
    if N <= dense_max:
        mu = np.sort(np.linalg.eigvalsh(A.toarray()))[::-1]
        mu2, mu_min = mu[1], mu[-1]
    else:
        top = spla.eigsh(A, k=3, which="LA", return_eigenvectors=False)
        bot = spla.eigsh(A, k=1, which="SA", return_eigenvectors=False)
        mu2, mu_min = np.sort(top)[::-1][1], bot[0]
    return dict(mu2=float(mu2), mu_min=float(mu_min), gap=float(3 - mu2),
                ramanujan_adj=bool(max(abs(mu2), abs(mu_min)) <= 2 * math.sqrt(2) + 1e-9))


def nonbacktracking_spectrum(surf: BMSurface, dense_max=3600):
    B = nonbacktracking(surf)
    D = B.shape[0]
    if D <= dense_max:
        rho = np.linalg.eigvals(B.toarray())
        mods = np.sort(np.abs(rho))[::-1]
    else:
        rho = spla.eigs(B, k=8, which="LM", return_eigenvectors=False)
        mods = np.sort(np.abs(rho))[::-1]
    # Perron eigenvalue is 2 (spectral radius); drop it (and a possible -2 for bipartite)
    nontrivial = mods[mods < 2 - 1e-8]
    rho2 = float(nontrivial[0]) if len(nontrivial) else 0.0
    return dict(nb_rho2=rho2, ramanujan_nb=bool(rho2 <= math.sqrt(2) + 1e-9),
                nb_top_moduli=[float(x) for x in mods[:6]])


# ----------------------------------------------------------------------------
# closed non-backtracking walks, cycles, cutting-sequence words
# ----------------------------------------------------------------------------
_L = ((1, 1), (0, 1))
_R = ((1, 0), (1, 1))


def _mul(a, b):
    return ((a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]),
            (a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]))


def _canonical_and_primitive(darts, d0):
    """darts is a closed walk starting at its minimal dart d0.  Keep it iff it is
    the lexicographically smallest rotation among rotations starting at d0 (so each
    cyclic walk is produced once) and it is not a power of a shorter walk."""
    ell = len(darts)
    for i in range(1, ell):
        if darts[i] == d0 and darts[i:] + darts[:i] < darts:
            return False
    for p in range(1, ell // 2 + 1):
        if ell % p == 0 and all(darts[i] == darts[i + p] for i in range(ell - p)):
            return False
    return True


def closed_nb_walks(surf: BMSurface, W: int):
    """Enumerate primitive closed non-backtracking walks of length <= W, each cyclic
    walk exactly once (canonical start = smallest exit dart, smallest rotation).
    Yields (darts, turns) with turns[i] in {1, 2}: exit side = entry side + turn (mod 3).
    Walks may revisit darts (that is what distinguishes tangles from simple cycles)."""
    tau = surf.tau
    for d0 in range(6 * surf.n):
        stack = [(d0, [d0], [])]              # current exit dart, dart list, turn list
        while stack:
            d, darts, turns = stack.pop()
            entry = tau[d]                    # entry dart of the next triangle
            v, s_in = entry // 3, entry % 3
            for t in (1, 2):
                dn = 3 * v + (s_in + t) % 3
                if dn < d0:
                    continue
                if dn == d0 and _canonical_and_primitive(darts, d0):
                    yield darts, turns + [t]
                if len(darts) < W:
                    stack.append((dn, darts + [dn], turns + [t]))


def cycle_and_length_features(surf: BMSurface, W: int = 10):
    """Girth, simple-cycle counts, tangle proxy and the combinatorial length spectrum
    (complete below ell_cut = 2 arccosh((W+2)/2))."""
    simple = Counter()      # length -> number of simple cycles (unoriented)
    nonsimple = Counter()   # length -> closed NB walks that revisit a vertex (unoriented)
    lengths = []            # geodesic lengths (unoriented, each once)
    words = Counter()       # cyclic word statistics: (#L, #R) -> count
    faces_seen = 0
    for darts, turns in closed_nb_walks(surf, W):
        ell = len(darts)
        verts = [d // 3 for d in darts]
        if len(set(verts)) == ell:
            simple[ell] += 1
        else:
            nonsimple[ell] += 1
        M = ((1, 0), (0, 1))
        for t in turns:
            M = _mul(M, _L if t == 1 else _R)
        tr = M[0][0] + M[1][1]
        if all(t == turns[0] for t in turns):     # once around a face = cusp: parabolic
            assert tr == 2
            faces_seen += 1
            continue
        assert tr > 2
        lengths.append(2.0 * math.acosh(tr / 2.0))
        words[(turns.count(1), turns.count(2))] += 1
    # every unoriented object was seen twice (both orientations)
    simple = {k: v // 2 for k, v in sorted(simple.items())}
    nonsimple = {k: v // 2 for k, v in sorted(nonsimple.items())}
    lengths = np.sort(np.array(lengths))[::2] if lengths else np.array([])
    ell_cut = 2.0 * math.acosh((W + 2) / 2.0)
    girth = min(simple) if simple else None
    # cusps of length <= W must have been seen once per orientation... (faces of the ribbon graph)
    n_faces_short = int(sum(1 for k in surf.k if k <= W))
    assert faces_seen == 2 * n_faces_short, (faces_seen, n_faces_short)
    return dict(
        W=W, ell_cut=ell_cut, girth=girth,
        simple_cycles={int(k): int(v) for k, v in simple.items()},
        n_short_cycles=int(sum(simple.values())),
        tangle_walks={int(k): int(v) for k, v in nonsimple.items()},
        n_tangle_walks=int(sum(nonsimple.values())),
        systole=(float(lengths[0]) if len(lengths) else None),
        n_geodesics_below_cut=int(len(lengths)),
        n_geodesics_below_2=int((lengths < 2.0).sum()),
        n_geodesics_below_3=int((lengths < 3.0).sum()),
        length_spectrum=[float(x) for x in lengths[:200]],
        word_types={"%d,%d" % k: int(v // 2) for k, v in sorted(words.items())},
    )


def cusp_features(surf: BMSurface):
    k = np.sort(surf.k)[::-1]
    return dict(V=int(surf.V), genus=int(surf.g), cusp_lengths=k.tolist(),
                k_max_fraction=float(k[0] / (6 * surf.n)),
                n_cusps_len1=int((k == 1).sum()), n_cusps_len2=int((k == 2).sum()),
                n_cusps_short=int((k <= 4).sum()))


def graph_features(surf: BMSurface, W: int = 10, spectra: bool = True) -> dict:
    f = dict(n=surf.n)
    f.update(cusp_features(surf))
    f.update(cycle_and_length_features(surf, W))
    if spectra:
        f.update(adjacency_spectrum(surf))
        f.update(nonbacktracking_spectrum(surf))
    return f
