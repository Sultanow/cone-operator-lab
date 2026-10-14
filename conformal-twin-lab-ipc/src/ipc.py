"""IPC-style contact handling (Li et al. 2020, 'Incremental Potential Contact'):
point-triangle and edge-edge barrier pairs below dhat, and additive/conservative CCD that
bounds every step so that no primitive pair can pass through another.  Incident primitives
(sharing a vertex) are excluded; dhat is a small fraction of the edge length, so non-incident
neighbours are never inside the barrier zone of a regular mesh."""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

def pt_dist(P, A, B, C):
    """pairwise unsigned distance point P[i] to triangle (A[i],B[i],C[i])"""
    ab, ac, ap = B - A, C - A, P - A
    d1 = (ab * ap).sum(-1); d2 = (ac * ap).sum(-1)
    bp = P - B; d3 = (ab * bp).sum(-1); d4 = (ac * bp).sum(-1)
    cp = P - C; d5 = (ab * cp).sum(-1); d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    den = va + vb + vc; den = np.where(np.abs(den) < 1e-300, 1e-300, den)
    Q = A + ab * (vb / den)[:, None] + ac * (vc / den)[:, None]
    def put(m, val):
        nonlocal Q; Q = np.where(m[:, None], val, Q)
    t = d4 - d3; s = d5 - d6
    put((va <= 0) & (t >= 0) & (s >= 0), B + (C - B) * (t / np.where(t + s == 0, 1, t + s))[:, None])
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), A + ac * (d2 / np.where(d2 - d6 == 0, 1, d2 - d6))[:, None])
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), A + ab * (d1 / np.where(d1 - d3 == 0, 1, d1 - d3))[:, None])
    put((d6 >= 0) & (d5 <= d6), C); put((d3 >= 0) & (d4 <= d3), B); put((d1 <= 0) & (d2 <= 0), A)
    return np.linalg.norm(P - Q, axis=-1)

def ee_dist(P1, Q1, P2, Q2):
    """pairwise distance between segments [P1,Q1] and [P2,Q2] (Ericson, robust clamping)"""
    d1, d2, r = Q1 - P1, Q2 - P2, P1 - P2
    a = (d1 * d1).sum(-1); e = (d2 * d2).sum(-1); f = (d2 * r).sum(-1); c = (d1 * r).sum(-1); b = (d1 * d2).sum(-1)
    den = a * e - b * b
    s = np.where(den > 1e-14 * a * e, np.clip((b * f - c * e) / np.where(den == 0, 1, den), 0, 1), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((b - c) / a, 0, 1), s))
    t = np.clip(t, 0, 1)
    return np.linalg.norm(P1 + d1 * s[:, None] - (P2 + d2 * t[:, None]), axis=-1)

class Contact:
    def __init__(self, F, E, nV):
        self.F, self.E, self.nV = F, E, nV
        rows = np.repeat(np.arange(len(F)), 3)
        self.VF = sp.csr_matrix((np.ones(3 * len(F), bool), (F.ravel(), rows)), shape=(nV, len(F)))   # vertex in face
    def _pt_candidates(self, X, R):
        F = self.F; cen = X[F].mean(1); rad = np.linalg.norm(X[F] - cen[:, None], axis=2).max(1)
        lists = cKDTree(cen).query_ball_point(X, R + rad.max())
        vi = np.repeat(np.arange(len(X)), [len(l) for l in lists]); fj = np.fromiter((f for l in lists for f in l), int, len(vi))
        if len(vi) == 0: return vi, fj
        keep = ~np.asarray(self.VF[vi, fj]).ravel().astype(bool)
        return vi[keep], fj[keep]
    def _ee_candidates(self, X, R):
        E = self.E; mid = (X[E[:, 0]] + X[E[:, 1]]) / 2; hl = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1).max() / 2
        P = cKDTree(mid).query_pairs(R + 2 * hl, output_type='ndarray')
        if len(P) == 0: return np.zeros(0, int), np.zeros(0, int)
        a, b = P[:, 0], P[:, 1]
        share = (E[a, 0] == E[b, 0]) | (E[a, 0] == E[b, 1]) | (E[a, 1] == E[b, 0]) | (E[a, 1] == E[b, 1])
        return a[~share], b[~share]
    def pairs(self, X, R):
        """all non-incident PT and EE pairs with distance < R"""
        vi, fj = self._pt_candidates(X, R); F = self.F
        if len(vi):
            d = pt_dist(X[vi], X[F[fj, 0]], X[F[fj, 1]], X[F[fj, 2]]); k = d < R; vi, fj = vi[k], fj[k]
        ea, eb = self._ee_candidates(X, R); E = self.E
        if len(ea):
            d = ee_dist(X[E[ea, 0]], X[E[ea, 1]], X[E[eb, 0]], X[E[eb, 1]]); k = d < R; ea, eb = ea[k], eb[k]
        return vi, fj, ea, eb
    def min_distance(self, X, R):
        vi, fj, ea, eb = self.pairs(X, R); F, E = self.F, self.E; m = R
        if len(vi): m = min(m, pt_dist(X[vi], X[F[fj, 0]], X[F[fj, 1]], X[F[fj, 2]]).min())
        if len(ea): m = min(m, ee_dist(X[E[ea, 0]], X[E[ea, 1]], X[E[eb, 0]], X[E[eb, 1]]).min())
        return m
    def ccd(self, X, D, s=0.2, iters=80):
        """largest t in [0,1] such that moving X -> X + t D keeps every non-incident pair apart
        (conservative advancement; each pair keeps at least s * its initial distance)."""
        L = np.linalg.norm(D, axis=1).max()
        if L == 0: return 1.0
        vi, fj, ea, eb = self.pairs(X, 2.0 * L * 1.01)
        F, E = self.F, self.E; tmin = 1.0
        for kind in ('pt', 'ee'):
            if kind == 'pt':
                if not len(vi): continue
                idx = np.stack([vi, F[fj, 0], F[fj, 1], F[fj, 2]], 1); dist = lambda Y: pt_dist(Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3])
                grp = [[0], [1, 2, 3]]
            else:
                if not len(ea): continue
                idx = np.stack([E[ea, 0], E[ea, 1], E[eb, 0], E[eb, 1]], 1); dist = lambda Y: ee_dist(Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3])
                grp = [[0, 1], [2, 3]]
            x, p = X[idx], D[idx]; p = p - p.mean(1, keepdims=True)
            n = np.linalg.norm(p, axis=2); lp = n[:, grp[0]].max(1) + n[:, grp[1]].max(1)
            d0 = dist(x); t = np.zeros(len(d0)); act = lp > 0; d = d0.copy()
            for _ in range(iters):
                if not act.any(): break
                step = np.where(act, np.minimum(0.9 * d / np.where(lp > 0, lp, 1), 1 - t), 0)
                t = t + step; d = dist(x + t[:, None, None] * p)
                act = act & (t < 1) & (d > s * d0)
            tmin = min(tmin, float(t.min()))
        return tmin
