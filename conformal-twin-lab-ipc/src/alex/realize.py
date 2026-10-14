"""Realisations of the intrinsic metric of Mizhaev's polyhedron P.
A realisation is X in R^{24x3} with |X_i - X_j|^2 = D_ij for the 84 edges of the geodesic
triangulation K (D_ij are integers, since P has integer coordinates). Every such X is a
triangulated surface isometric to P (its triangles are congruent to those of P); nonagons may
fold along diagonals. Trivial solutions: rigid motions and reflections of P."""
import json, os, numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results.json')))
P = np.array(R['vertices'], float); E = np.array(R['edges']) - 1
TRI = np.array(R['triangles']) - 1; FACES = [[x - 1 for x in f] for f in R['faces']]
PERM = np.array(R['symmetry_perm']) - 1
D = np.array([int(round(((P[i] - P[j]) ** 2).sum())) for i, j in E], float)   # exact integers
SCALE = np.sqrt(D.mean())

def random_triangulation(rng):
    """random triangulation of every nonagon by random ear clipping in its own plane (diagonals stay inside
    the possibly non-convex polygon); returns triangles and edges with exact squared lengths"""
    tris = []
    for f in FACES:
        Q = P[f] - P[f].mean(0); _, _, Vt = np.linalg.svd(Q); xy = Q @ Vt[:2].T
        area = 0.5 * sum(xy[i, 0] * xy[(i + 1) % 9, 1] - xy[(i + 1) % 9, 0] * xy[i, 1] for i in range(9)); sgn = np.sign(area)
        idx = list(range(9))
        while len(idx) > 3:
            ears = []
            for k in range(len(idx)):
                a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
                cr = (xy[b, 0] - xy[a, 0]) * (xy[c, 1] - xy[a, 1]) - (xy[b, 1] - xy[a, 1]) * (xy[c, 0] - xy[a, 0])
                if cr * sgn <= 1e-9: continue
                def inside(p):
                    d1 = (xy[b, 0] - xy[a, 0]) * (p[1] - xy[a, 1]) - (xy[b, 1] - xy[a, 1]) * (p[0] - xy[a, 0])
                    d2 = (xy[c, 0] - xy[b, 0]) * (p[1] - xy[b, 1]) - (xy[c, 1] - xy[b, 1]) * (p[0] - xy[b, 0])
                    d3 = (xy[a, 0] - xy[c, 0]) * (p[1] - xy[c, 1]) - (xy[a, 1] - xy[c, 1]) * (p[0] - xy[c, 0])
                    # closed triangle: a vertex lying exactly ON the would-be diagonal also blocks it
                    # (P has such collinear triples, e.g. vertices 5, 8, 7 of the first nonagon)
                    return (d1 * sgn > -1e-9) and (d2 * sgn > -1e-9) and (d3 * sgn > -1e-9)
                if any(inside(xy[o]) for o in idx if o not in (a, b, c)): continue
                ears.append(k)
            k = ears[rng.integers(len(ears))]; a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            tris.append((f[a], f[b], f[c])); idx.pop(k)
        tris.append(tuple(f[i] for i in idx))
    tris = np.array(tris); es = sorted({tuple(sorted(e)) for t in tris for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))})
    return tris, np.array(es)

def use_edges(edges):
    """switch the module to another geodesic triangulation of the same flat metric"""
    global E, D, SCALE
    E = np.array(edges); D = np.array([int(round(((P[i] - P[j]) ** 2).sum())) for i, j in E], float); SCALE = np.sqrt(D.mean())
def resid(x):
    X = x.reshape(-1, 3); d = X[E[:, 0]] - X[E[:, 1]]
    return ((d * d).sum(1) - D) / SCALE ** 2
def jac(x):
    X = x.reshape(-1, 3); d = X[E[:, 0]] - X[E[:, 1]]; J = np.zeros((len(E), X.size))
    for c in range(3):
        J[np.arange(len(E)), 3 * E[:, 0] + c] = 2 * d[:, c]; J[np.arange(len(E)), 3 * E[:, 1] + c] = -2 * d[:, c]
    return J / SCALE ** 2
def solve(X0, tol=1e-15):
    r = least_squares(resid, X0.ravel(), jac=jac, method='lm', xtol=tol, ftol=tol, gtol=tol, max_nfev=4000)
    return r.x.reshape(-1, 3), float(np.sqrt(np.mean(r.fun ** 2)))
def align(X, Y):
    """RMSD of X to Y after the best proper or improper rigid motion (Kabsch); returns rmsd, aligned X"""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0); best = (np.inf, None)
    for refl in (1, -1):
        Xr = Xc * np.array([1, 1, refl]); U, S, Vt = np.linalg.svd(Xr.T @ Yc)
        d = np.sign(np.linalg.det(U @ Vt)); Rm = U @ np.diag([1, 1, d]) @ Vt; A = Xr @ Rm
        rmsd = np.sqrt(((A - Yc) ** 2).sum(1).mean())
        if rmsd < best[0]: best = (rmsd, A + Y.mean(0))
    return best
def shape_distance(X, Y):
    """distance between two labelled realisations modulo congruence and the relabelling symmetry tau"""
    best = np.inf; idx = np.arange(24)
    for k in range(4):
        best = min(best, align(X[idx], Y)[0]); idx = PERM[idx]
    return best
def planarity(X):
    """max over nonagons of the largest distance of a vertex from the best-fit plane"""
    m = 0
    for f in FACES:
        Q = X[f] - X[f].mean(0); s = np.linalg.svd(Q, compute_uv=False); m = max(m, s[-1] / np.sqrt(len(f)))
    return m
