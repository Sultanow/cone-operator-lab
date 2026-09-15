"""
bmsurf.py -- Brooks-Makover random surfaces and their hyperbolic FEM mesh.

Model
-----
2n ideal hyperbolic triangles are glued along a random cubic ribbon graph
(configuration model, orientation = the fixed counter-clockwise order of the
three sides).  Gluings are zero-shear (edge "midpoints" = tangency points of
the height-1 horocycles are identified), so the result is a complete finite-
area hyperbolic surface S with V cusps and genus  g = n/2 + 1 - V/2.

Charts
------
Every triangle is the standard ideal triangle T = (0, 1, inf) of the upper
half-plane. Sides (ccw): 0 = geodesic 0->1 (semicircle), 1 = 1->inf (x=1),
2 = inf->0 (x=0).  Corner s = the corner between side s and side s+1; its
ideal vertex is (1, inf, 0)[s].  Corner s has its own "corner chart" in which
that vertex sits at inf and the corner region is the strip 0<=x<=1, y>=1;
the maps to T are  s=1: z,  s=2: 1/(1-z),  s=0: 1-1/z  (rotations of T).

Mesh
----
T is cut along the three height-1 horocycles into a compact central region
(area pi-3) and three corner strips {0<=x<=1, 1<=y<=Y_c}.  Each cusp c with
k_c corners is truncated at the horocycle of length L_c ~ L0 (Y_c = k_c/L_c),
which is also where the conformal cap disc |w| <= r0 = exp(-2 pi / L_c),
w = exp(2 pi i (x_c + i y)/k_c), is attached for the compactification.

Because the Dirichlet energy is conformally invariant in 2D, every element is
assembled with the *Euclidean* P1 stiffness in its own chart; the metric enters
only through the area density rho_0 (1/y^2 in the triangle charts, the
constant (L_c/(2 pi r0))^2 in the cap charts).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    import triangle as _triangle
except ImportError:  # pragma: no cover
    _triangle = None


# ----------------------------------------------------------------------------
# Combinatorics
# ----------------------------------------------------------------------------
class BMSurface:
    """Random Brooks-Makover surface built from 2n ideal triangles.

    Darts are d = 3 v + s (triangle v, side s). tau is the fixed-point-free
    involution pairing sides, sigma(3v+s) = 3v+(s+1)%3 is the ccw rotation.
    Corner s of triangle v is identified with dart 3v+s.
    """

    def __init__(self, n: int, tau: np.ndarray):
        self.n = int(n)
        self.tau = np.asarray(tau, dtype=int)
        N = 6 * self.n
        assert self.tau.shape == (N,) and (self.tau[self.tau] == np.arange(N)).all()
        assert (self.tau != np.arange(N)).all()
        self.sigma = np.array([3 * (d // 3) + (d % 3 + 1) % 3 for d in range(N)])
        # cusps = orbits of d -> tau(sigma(d))  (faces of the ribbon graph)
        cusp_of = -np.ones(N, dtype=int)
        cusps = []
        for d0 in range(N):
            if cusp_of[d0] >= 0:
                continue
            cyc, d = [], d0
            while cusp_of[d] < 0:
                cusp_of[d] = len(cusps)
                cyc.append(d)
                d = self.tau[self.sigma[d]]
            cusps.append(cyc)
        self.cusps = cusps
        self.cusp_of = cusp_of
        self.k = np.array([len(c) for c in cusps])          # cusp lengths (corners)
        # cusp coordinate offset of each corner: offset decreases along the orbit
        self.offset = np.zeros(N, dtype=int)
        for cyc in cusps:
            kk = len(cyc)
            for j, d in enumerate(cyc):
                self.offset[d] = (-j) % kk
        self.V = len(cusps)
        self.chi = self.V - self.n                       # V - E + F = V - 3n + 2n
        assert self.chi % 2 == 0
        self.g = (2 - self.chi) // 2

    # -- helpers -----------------------------------------------------------
    def edge_id(self, d: int) -> int:
        return min(d, self.tau[d])

    def is_primary(self, d: int) -> bool:
        return d <= self.tau[d]

    def is_connected(self) -> bool:
        parent = list(range(2 * self.n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for d in range(6 * self.n):
            a, b = find(d // 3), find(self.tau[d] // 3)
            if a != b:
                parent[a] = b
        return len({find(v) for v in range(2 * self.n)}) == 1

    @staticmethod
    def random(n: int, rng: np.random.Generator, connected: bool = True):
        """Configuration-model pairing of the 6n darts; resample until connected."""
        attempts = 0
        while True:
            attempts += 1
            perm = rng.permutation(6 * n)
            tau = np.empty(6 * n, dtype=int)
            tau[perm[0::2]] = perm[1::2]
            tau[perm[1::2]] = perm[0::2]
            surf = BMSurface(n, tau)
            if (not connected) or surf.is_connected():
                surf.attempts = attempts
                return surf


# ----------------------------------------------------------------------------
# Mesh
# ----------------------------------------------------------------------------
@dataclass
class CuspChain:
    """Nodes on the truncation horocycle of one cusp, in cyclic order."""
    cusp: int
    k: int
    Y: float
    L: float                      # horocycle length k/Y  (~ L0)
    r0: float                     # cap radius exp(-2 pi / L)
    node: np.ndarray              # global node ids, increasing cusp coordinate
    x: np.ndarray                 # cusp coordinates in [0, k)


@dataclass
class Mesh:
    nnodes: int
    tri: np.ndarray               # (E,3) global node ids
    xy: np.ndarray                # (E,3,2) chart coordinates of the vertices
    kind: np.ndarray              # (E,) 0: hyperbolic chart (rho_0 = 1/y^2), 1: flat cap chart
    rho_const: np.ndarray         # (E,) density of g_0 for kind==1 elements
    tag: np.ndarray               # (E,) 0: S_Y (strips+central), 1: cusp extension, 2: cap
    cap_chains: list = field(default_factory=list)   # horocycles of length ~L0 (cap boundary)
    dtn_chains: list = field(default_factory=list)   # horocycles at Y*exp(T_ext) (DtN boundary)
    central_nodes: np.ndarray = None
    cap_interior_nodes: np.ndarray = None
    ext_nodes: np.ndarray = None
    thick_nodes: np.ndarray = None
    height1_rows: dict = field(default_factory=dict)  # cusp -> node ids on its height-1 horocycle
    info: dict = field(default_factory=dict)

    def nodes_S(self) -> np.ndarray:
        """Nodes of the cusped surface model (S_Y + cusp extensions)."""
        mask = np.ones(self.nnodes, dtype=bool)
        mask[self.cap_interior_nodes] = False
        return np.nonzero(mask)[0]

    def nodes_Sbar(self) -> np.ndarray:
        """Nodes of the compactification model (S_Y + caps)."""
        mask = np.ones(self.nnodes, dtype=bool)
        mask[self.ext_nodes] = False
        return np.nonzero(mask)[0]

    def nodes_SY(self) -> np.ndarray:
        mask = np.ones(self.nnodes, dtype=bool)
        mask[self.ext_nodes] = False
        mask[self.cap_interior_nodes] = False
        return np.nonzero(mask)[0]


class _Registry:
    def __init__(self):
        self.idx = {}

    def get(self, key):
        i = self.idx.get(key)
        if i is None:
            i = len(self.idx)
            self.idx[key] = i
        return i

    def __len__(self):
        return len(self.idx)


def _corner_to_T(s: int, z: np.ndarray) -> np.ndarray:
    """Corner chart of corner s -> chart T."""
    if s == 1:
        return z
    if s == 2:
        return 1.0 / (1.0 - z)
    return 1.0 - 1.0 / z


def _merge_rows(bot, top):
    """Triangulate the band between two node rows (ids, xs, y), both spanning
    the same x-range with increasing xs.  Returns [(id,x,y)*3] ccw triangles."""
    bid, bx, by = bot
    tid, tx, ty = top
    tris, i, j = [], 0, 0
    nb, nt = len(bid) - 1, len(tid) - 1
    while i < nb or j < nt:
        if j == nt or (i < nb and bx[i + 1] <= tx[j + 1]):
            tris.append(((bid[i], bx[i], by), (bid[i + 1], bx[i + 1], by), (tid[j], tx[j], ty)))
            i += 1
        else:
            tris.append(((bid[i], bx[i], by), (tid[j + 1], tx[j + 1], ty), (tid[j], tx[j], ty)))
            j += 1
    return tris


def build_mesh(surf: BMSurface, h: float = 0.1, L0: float = 1.0, T_ext: float = 6.0,
               chain_min_nodes: int = 24, central_opts: str | None = None) -> Mesh:
    """Build the P1 mesh: truncated surface S_Y, cusp extensions (for the exact cusp
    DtN of the cusped surface) and conformal caps (for the compactification).

    h          hyperbolic mesh size
    L0         length of the truncation horocycle where the caps are attached
    T_ext      hyperbolic depth of the single-column cusp extension beyond that horocycle
    """
    if _triangle is None:
        raise RuntimeError("pip install triangle  (J. Shewchuk's mesher) is required")
    n = surf.n
    reg = _Registry()
    tri, xy, kind, rho_c, tag = [], [], [], [], []
    near_top, central_nodes, cap_interior, ext_nodes = set(), set(), set(), set()
    ntop = int(math.ceil(1.0 / h))

    J = np.array([max(0, int(round(math.log(k / L0) / h))) for k in surf.k])
    Y = np.exp(J * h)
    L = surf.k / Y
    Jext = int(round(T_ext / h))

    def m_of(j):
        return max(1, int(round(math.exp(-j * h) / h)))

    def edge_key(v, side, toward_end, j):
        if j == 0:
            return ("E", surf.edge_id(3 * v + side), 0, 0)
        d = 3 * v + side
        e = 1 if toward_end else 0
        if not surf.is_primary(d):
            e = 1 - e
        return ("E", surf.edge_id(d), e, j)

    def add(tris, k_, rc, tg):
        for t in tris:
            tri.append([p[0] for p in t])
            xy.append([(p[1], p[2]) for p in t])
            kind.append(k_); rho_c.append(rc); tag.append(tg)

    # ---- corner strips + cusp extensions ------------------------------------------
    bottom_rows, top_rows, dtn_rows = {}, {}, {}
    for d in range(6 * n):
        v, s = divmod(d, 3)
        c = surf.cusp_of[d]
        Jc, kc = int(J[c]), int(surf.k[c])
        rows = []
        for j in range(Jc + 1 + Jext):
            m = m_of(j)
            if j == Jc:
                m = max(m, int(math.ceil(chain_min_nodes / kc)))
            xs = np.arange(m + 1) / m
            ids = []
            for i in range(m + 1):
                if i == 0:
                    key = edge_key(v, (s + 1) % 3, False, j)      # left edge = side s+1
                elif i == m:
                    key = edge_key(v, s, True, j)                 # right edge = side s
                else:
                    key = ("S", d, j, i)
                ids.append(reg.get(key))
            if Jc - ntop < j <= Jc:
                near_top.update(ids)
            if j > Jc:
                ext_nodes.update(ids)
            rows.append((ids, xs, math.exp(j * h)))
        for j in range(Jc):
            add(_merge_rows(rows[j], rows[j + 1]), 0, 0.0, 0)
        for j in range(Jc, Jc + Jext):
            add(_merge_rows(rows[j], rows[j + 1]), 0, 0.0, 1)
        bottom_rows[d] = rows[0]
        top_rows[d] = rows[Jc]
        dtn_rows[d] = rows[Jc + Jext]

    # ---- central regions (chart T) ---------------------------------------------------
    A_central = 0.45 * (0.8 * h) ** 2
    opts = central_opts or ("pYq20a%.6g" % A_central)
    for v in range(2 * n):
        pts, ids = [], []
        for s in (1, 2, 0):
            rid, rx, _ = bottom_rows[3 * v + s]
            z = _corner_to_T(s, np.asarray(rx) + 1j)
            order = range(len(rid) - 1, -1, -1)
            if ids:
                assert rid[len(rid) - 1] == ids[-1]
                order = range(len(rid) - 2, -1, -1)
            for i in order:
                ids.append(rid[i]); pts.append((z[i].real, z[i].imag))
        assert ids[0] == ids[-1]
        ids, pts = ids[:-1], pts[:-1]
        P = np.array(pts); m = len(P)
        segs = np.column_stack([np.arange(m), (np.arange(m) + 1) % m])
        area2 = np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
        assert area2 > 0, "central polygon not ccw"
        out = _triangle.triangulate({"vertices": P, "segments": segs}, opts)
        Vo, To = out["vertices"], out["triangles"]
        assert np.allclose(Vo[:m], P), "triangle reordered input vertices"
        gid = list(ids) + [reg.get(("C", v, i)) for i in range(m, len(Vo))]
        central_nodes.update(gid)
        tris = [tuple((gid[q], Vo[q, 0], Vo[q, 1]) for q in map(int, t)) for t in To]
        add(tris, 0, 0.0, 0)

    # ---- chains + caps ------------------------------------------------------------------
    def chain(rows_of, c, cyc, Yc):
        k = len(cyc)
        by_offset = {surf.offset[d]: d for d in cyc}
        node, xc = [], []
        for o in range(k):
            rid, rx, _ = rows_of[by_offset[o]]
            nxt = by_offset[(o + 1) % k]
            assert rid[-1] == rows_of[nxt][0][0], "cusp chain mismatch"
            node.extend(rid[:-1]); xc.extend(o + np.asarray(rx[:-1]))
        Lc = k / Yc
        return CuspChain(c, k, float(Yc), float(Lc), math.exp(-2 * math.pi / Lc),
                         np.array(node), np.array(xc, dtype=float))

    cap_chains, dtn_chains, height1 = [], [], {}
    from scipy.spatial import Delaunay
    for c, cyc in enumerate(surf.cusps):
        cap_chains.append(chain(top_rows, c, cyc, Y[c]))
        dtn_chains.append(chain(dtn_rows, c, cyc, Y[c] * math.exp(Jext * h)))
        height1[c] = sorted({i for d in cyc for i in bottom_rows[d][0]})
        ch = cap_chains[-1]
        Nb, r0, k = len(ch.node), ch.r0, ch.k
        th = 2 * math.pi * ch.x / k
        P = [np.column_stack([r0 * np.cos(th), r0 * np.sin(th)])]
        gid = list(ch.node)
        R = 3 if Nb >= 8 else 2
        for r in range(1, R):
            nr = max(4, int(round(Nb * (R - r) / R)))
            tr = 2 * math.pi * (np.arange(nr) + 0.5) / nr
            rr = r0 * (R - r) / R
            P.append(np.column_stack([rr * np.cos(tr), rr * np.sin(tr)]))
            gid.extend(reg.get(("K", c, r, i)) for i in range(nr))
        P.append(np.zeros((1, 2))); gid.append(reg.get(("K", c, 0, 0)))
        cap_interior.update(gid[Nb:])
        P = np.vstack(P)
        dens_c = (ch.L / (2 * math.pi * r0)) ** 2
        tris = []
        for t in Delaunay(P).simplices:
            q = [int(a) for a in t]
            a2 = (P[q[1], 0] - P[q[0], 0]) * (P[q[2], 1] - P[q[0], 1]) - (P[q[2], 0] - P[q[0], 0]) * (P[q[1], 1] - P[q[0], 1])
            if a2 < 0:
                q[1], q[2] = q[2], q[1]
            tris.append(tuple((gid[a], P[a, 0], P[a, 1]) for a in q))
        add(tris, 1, dens_c, 2)

    tri = np.array(tri, dtype=int); xy = np.array(xy, dtype=float)
    kind = np.array(kind, dtype=int); rho_c = np.array(rho_c, dtype=float); tag = np.array(tag, dtype=int)
    nn = len(reg)
    thick = np.ones(nn, dtype=bool)
    for S_ in (near_top, cap_interior, ext_nodes):
        thick[list(S_)] = False
    return Mesh(nn, tri, xy, kind, rho_c, tag, cap_chains, dtn_chains,
                central_nodes=np.array(sorted(central_nodes)),
                cap_interior_nodes=np.array(sorted(cap_interior)),
                ext_nodes=np.array(sorted(ext_nodes)),
                thick_nodes=np.nonzero(thick)[0],
                height1_rows=height1,
                info=dict(h=h, L0=L0, T_ext=T_ext, J=J.tolist(), Y=Y.tolist(), L=L.tolist(),
                          n_elements=int(len(tri)), n_ext=int((tag == 1).sum()), n_cap=int((tag == 2).sum())))
