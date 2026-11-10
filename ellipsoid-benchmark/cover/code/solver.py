"""Finite-difference Dirichlet eigenmodes of the triaxial ellipsoid
Omega(1, 1.5, 2.3), solved per symmetry class on one octant."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, time, itertools

A, B, C = 1.0, 1.5, 2.3

def octant_operator(h, par):
    nx, ny, nz = int(np.ceil(A/h)), int(np.ceil(B/h)), int(np.ceil(C/h))
    x = (np.arange(nx)+0.5)*h; y = (np.arange(ny)+0.5)*h; z = (np.arange(nz)+0.5)*h
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    inside = (X/A)**2 + (Y/B)**2 + (Z/C)**2 < 1.0
    idx = -np.ones(inside.shape, dtype=np.int64)
    idx[inside] = np.arange(inside.sum())
    n = inside.sum()
    I, J, V = [], [], []
    diag = np.full(n, 6.0)
    pts = np.argwhere(inside)
    me = idx[inside]
    for ax in range(3):
        for s in (+1, -1):
            nb = pts.copy(); nb[:, ax] += s
            if s == -1:
                refl = nb[:, ax] < 0          # neighbour across symmetry plane
                diag[me[refl]] -= par[ax]     # ghost = par * self
                ok = ~refl
            else:
                ok = nb[:, ax] < inside.shape[ax]
            ok_idx = np.where(ok)[0]
            nbv = idx[nb[ok_idx, 0], nb[ok_idx, 1], nb[ok_idx, 2]]
            good = nbv >= 0                    # outside ellipsoid -> Dirichlet 0
            I.append(me[ok_idx[good]]); J.append(nbv[good]); V.append(-np.ones(good.sum()))
    I.append(me); J.append(me); V.append(diag)
    L = sp.csc_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(n, n)) / h**2
    return L, inside, idx, (x, y, z)

def modes(h, par, k, sigma):
    L, inside, idx, ax = octant_operator(h, par)
    vals, vecs = sla.eigsh(L, k=k, sigma=sigma, which='LM')
    o = np.argsort(vals)
    return vals[o], vecs[:, o], inside, ax

if __name__ == "__main__":
    t = time.time()
    L, inside, idx, ax = octant_operator(0.04, (1, 1, 1))
    print(L.shape, time.time()-t)
