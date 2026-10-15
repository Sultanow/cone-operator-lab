"""NumPy geometry helpers: closest points on the polyhedron, exact self-intersection test."""
import numpy as np
from scipy.spatial import cKDTree

def closest_on_triangles(Pt, A, B, C):
    """closest points from each point in Pt (m,3) to each triangle (k,3) -> returns (m,k,3)"""
    p = Pt[:, None, :]; a, b, c = A[None], B[None], C[None]
    ab, ac, ap = b - a, c - a, p - a
    d1 = (ab * ap).sum(-1); d2 = (ac * ap).sum(-1)
    bp = p - b; d3 = (ab * bp).sum(-1); d4 = (ac * bp).sum(-1)
    cp = p - c; d5 = (ab * cp).sum(-1); d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    den = np.where(np.abs(va + vb + vc) < 1e-30, 1e-30, va + vb + vc)
    v = vb / den; w = vc / den
    res = a + ab * v[..., None] + ac * w[..., None]                  # interior default
    def put(mask, val):
        nonlocal res
        res = np.where(mask[..., None], val, res)
    # edge regions (order matters: later puts override, so vertices go last)
    t = d4 - d3; s = d5 - d6
    m_bc = (va <= 0) & (t >= 0) & (s >= 0); put(m_bc, b + (c - b) * (t / np.where(t + s == 0, 1, t + s))[..., None])
    m_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0); put(m_ac, a + ac * (d2 / np.where(d2 - d6 == 0, 1, d2 - d6))[..., None])
    m_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0); put(m_ab, a + ab * (d1 / np.where(d1 - d3 == 0, 1, d1 - d3))[..., None])
    put((d6 >= 0) & (d5 <= d6), np.broadcast_to(c, res.shape))
    put((d3 >= 0) & (d4 <= d3), np.broadcast_to(b, res.shape))
    put((d1 <= 0) & (d2 <= 0), np.broadcast_to(a, res.shape))
    return res

def project_to_P(X, VP, F0, chunk=4096):
    A, B, C = VP[F0[:, 0]], VP[F0[:, 1]], VP[F0[:, 2]]
    N = np.cross(B - A, C - A); N /= np.linalg.norm(N, axis=1, keepdims=True)
    tgt = np.empty_like(X); nrm = np.empty_like(X); dist = np.empty(len(X))
    for s in range(0, len(X), chunk):
        Q = closest_on_triangles(X[s:s + chunk], A, B, C)
        d = np.linalg.norm(Q - X[s:s + chunk, None], axis=2); k = d.argmin(1)
        r = np.arange(len(k)); tgt[s:s + chunk] = Q[r, k]; nrm[s:s + chunk] = N[k]; dist[s:s + chunk] = d[r, k]
    return tgt, nrm, dist

def self_intersections(X, F):
    """number of pairs of triangles without a common vertex that intersect.
    Robust for non-generic input (integer, coplanar, parallel): the parallel test is scale-relative,
    and a segment lying in a triangle's plane is tested in 2D instead of being skipped."""
    tri = X[F]; cen = tri.mean(1); rad = np.linalg.norm(tri - cen[:, None], axis=2).max(1)
    tree = cKDTree(cen); pairs = np.array(sorted(tree.query_pairs(2 * rad.max())), dtype=int).reshape(-1, 2)
    if len(pairs) == 0: return 0
    i, j = pairs[:, 0], pairs[:, 1]
    keep = np.linalg.norm(cen[i] - cen[j], axis=1) < rad[i] + rad[j]; i, j = i[keep], j[keep]
    share = np.zeros(len(i), bool)
    for a in range(3):
        for b in range(3): share |= F[i, a] == F[j, b]
    i, j = i[~share], j[~share]
    if len(i) == 0: return 0
    def cross2(a, b): return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    def seg_tri(S0, S1, T):
        e1 = T[:, 1] - T[:, 0]; e2 = T[:, 2] - T[:, 0]; d = S1 - S0
        nrm = np.cross(e1, e2); scale = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(d, axis=1)
        h = np.cross(d, e2); a = np.einsum('ij,ij->i', e1, h)
        ok = np.abs(a) > 1e-10 * scale                                   # scale-relative parallel test
        f = np.where(ok, 1 / np.where(ok, a, 1), 0); s = S0 - T[:, 0]
        u = f * np.einsum('ij,ij->i', s, h); q = np.cross(s, e1)
        v = f * np.einsum('ij,ij->i', d, q); t = f * np.einsum('ij,ij->i', e2, q); eps = 1e-9
        hit = ok & (u > eps) & (v > eps) & (u + v < 1 - eps) & (t > eps) & (t < 1 - eps)
        # parallel segments that lie IN the plane of the triangle: 2D test (endpoint strictly inside, or proper edge crossing)
        nn = np.linalg.norm(nrm, axis=1); dist = np.abs(np.einsum('ij,ij->i', s, nrm)) / np.where(nn > 0, nn, 1)
        cop = (~ok) & (dist < 1e-9 * np.linalg.norm(e1, axis=1))
        if cop.any():
            k = np.where(cop)[0]; n_ = nrm[k] / nn[k][:, None]
            u1 = e1[k] / np.linalg.norm(e1[k], axis=1)[:, None]; u2 = np.cross(n_, u1)
            P2 = lambda Y: np.stack([np.einsum('ij,ij->i', Y - T[k, 0], u1), np.einsum('ij,ij->i', Y - T[k, 0], u2)], 1)
            A, B, C = P2(T[k, 0]), P2(T[k, 1]), P2(T[k, 2]); p0, p1 = P2(S0[k]), P2(S1[k])
            area = cross2(B - A, C - A); sg = np.sign(area); tol = 1e-9 * np.abs(area)
            def inside(p): return (sg * cross2(B - A, p - A) > tol) & (sg * cross2(C - B, p - B) > tol) & (sg * cross2(A - C, p - C) > tol)
            def proper(p, q, r, w):   # segments p-q and r-w cross in their interiors
                d1 = cross2(q - p, r - p); d2 = cross2(q - p, w - p); d3 = cross2(w - r, p - r); d4 = cross2(w - r, q - r)
                tt = 1e-9 * np.linalg.norm(q - p, axis=1) * np.linalg.norm(w - r, axis=1)
                return (d1 * d2 < -tt ** 2) & (d3 * d4 < -tt ** 2)
            h2 = inside(p0) | inside(p1) | proper(p0, p1, A, B) | proper(p0, p1, B, C) | proper(p0, p1, C, A)
            hit[k] |= h2
        return hit
    hit = np.zeros(len(i), bool)
    for A_, B_ in ((i, j), (j, i)):
        for a, b in ((0, 1), (1, 2), (2, 0)): hit |= seg_tri(tri[A_, a], tri[A_, b], tri[B_])
    return int(hit.sum())
