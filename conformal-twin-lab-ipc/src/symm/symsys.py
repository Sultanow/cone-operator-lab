"""C4-symmetric realisations of P's intrinsic metric: X[tau^m r_k] = T^m x_k for 6 orbit representatives.
Gauge: rotations about the z-axis commute with T, fixed by putting x_0 in the xz-plane (y-coordinate 0).
Unknowns: x_0 = (a, 0, c) and x_1..x_5 in R^3 -> 17 unknowns. Equations: one per tau-orbit of edges of K."""
import json, os, numpy as np
R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')))
P = np.array(R['vertices'], float); E = np.array(R['edges']) - 1; PERM = np.array(R['symmetry_perm']) - 1
T = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1.]]); Tp = [np.linalg.matrix_power(T, m) for m in range(4)]
S = 300.0                                               # length scale (unknowns are coordinates / 300)
orb = -np.ones(24, int); pw = np.zeros(24, int); reps = []
for v in range(24):
    if orb[v] >= 0: continue
    k = len(reps); reps.append(v); w = v
    for m in range(4): orb[w] = k; pw[w] = m; w = PERM[w]
assert all(np.allclose(P[v], Tp[pw[v]] @ P[reps[orb[v]]]) for v in range(24))
# edge orbits: |T^a x_k - T^b x_l|^2 = |x_k - T^(b-a) x_l|^2 ; canonical key (k, l, (b-a) mod 4) up to swapping
eqs = {}
for (i, j) in E:
    k, l, d = orb[i], orb[j], (pw[j] - pw[i]) % 4
    key = (k, l, d) if (k, l) <= (l, k) else (l, k, (-d) % 4)
    if k == l: key = (k, k, min(d, (-d) % 4))
    Dij = int(round(((P[i] - P[j]) ** 2).sum()))
    if key in eqs: assert eqs[key] == Dij
    eqs[key] = Dij
KEYS = list(eqs.keys()); DV = np.array([eqs[k] for k in KEYS], float) / S ** 2
def unpack(z):   # z: (..., 17) -> x: (..., 6, 3)
    x = np.zeros(z.shape[:-1] + (6, 3), dtype=z.dtype)
    x[..., 0, 0] = z[..., 0]; x[..., 0, 2] = z[..., 1]; x[..., 1:, :] = z[..., 2:].reshape(z.shape[:-1] + (5, 3))
    return x
def residuals(z):
    """all 22 orbit equations, vectorised over leading axes"""
    x = unpack(z); out = []
    for (k, l, d), Dv in zip(KEYS, DV):
        diff = x[..., k, :] - x[..., l, :] @ Tp[d].T
        out.append((diff * diff).sum(-1) - Dv)
    return np.stack(out, -1)
def jacobian(z):
    x = unpack(z); J = np.zeros(z.shape[:-1] + (len(KEYS), 6, 3), dtype=z.dtype)
    for q, (k, l, d) in enumerate(KEYS):
        diff = x[..., k, :] - x[..., l, :] @ Tp[d].T
        J[..., q, k, :] += 2 * diff
        J[..., q, l, :] += -2 * diff @ Tp[d]        # d/dx_l of |x_k - T^d x_l|^2 = -2 (T^d)^T diff
    Jx = J.reshape(z.shape[:-1] + (len(KEYS), 18))
    return np.concatenate([Jx[..., [0]], Jx[..., [2]], Jx[..., 3:]], -1)   # drop the gauge-fixed y of x_0
def z_of(X):
    """configuration (24x3, symmetric, rotated so that x_0 has y = 0) -> z"""
    x0 = X[reps[0]]; phi = np.arctan2(x0[1], x0[0]); c, s = np.cos(-phi), np.sin(-phi)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); Y = X @ Rz.T / S
    xr = Y[reps]; return np.r_[xr[0, 0], xr[0, 2], xr[1:].ravel()]
def X_of(z):
    x = unpack(z) * S; return np.array([Tp[pw[v]] @ x[orb[v]] for v in range(24)])
