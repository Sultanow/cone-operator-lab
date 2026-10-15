"""Table 3 of the paper: the exact way back P -> (h_P, 24 marked points, 24 cone angles) -> P on refinement
levels 0-3, plus the rebuild of P in R^3 from the recovered edge lengths on level 0.
Usage: python src/wpc/roundtrip.py OUTDIR      (writes OUTDIR/troyanov_roundtrip.json)"""
import os, sys, json, numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC); sys.path.insert(0, HERE)
from fine_mesh import build
from uniformize import uniformize
from troyanov import flat_from_hyperbolic
out = sys.argv[1] if len(sys.argv) > 1 else '.'; os.makedirs(out, exist_ok=True)
R = json.load(open(os.path.join(SRC, 'results.json'))); VP = np.array(R['vertices'], float)
theta = 2 * np.pi - np.array(R['original_curvature'])                       # cone angles of P
print(f'Gauss-Bonnet: sum of angle defects / 2pi = {(2 * np.pi - theta).sum() / (2 * np.pi):.12f} (chi = -4)')
rows = []
for lev in (0, 1, 2, 3):
    M = build(lev, SRC); U = uniformize(M, verbose=False); XP = M['XP']          # fine-mesh vertex numbering
    marked = [int(np.argmin(np.linalg.norm(XP - VP[i], axis=1))) for i in range(24)]
    lP = np.linalg.norm(XP[M['edges'][:, 0]] - XP[M['edges'][:, 1]], axis=1)
    l, res, it, info = flat_from_hyperbolic(M, U['L'], marked, theta)
    s = np.exp(np.mean(np.log(lP / l))); err = float(np.abs(s * l / lP - 1).max())
    row = dict(level=lev, V=len(XP), forward=U['residual'], inverse=res, newton_steps=it, length_error=err, **info)
    if lev == 0:                       # rebuild P in R^3 from the 84 recovered lengths, perturbed starts
        Dk = (s * l) ** 2; E0 = M['edges']; rng = np.random.default_rng(0); devs = []
        def rr(x): X = x.reshape(-1, 3); d = X[E0[:, 0]] - X[E0[:, 1]]; return ((d * d).sum(1) - Dk) / Dk.mean()
        for trial in range(40):
            sol = least_squares(rr, (XP + 20 * rng.normal(size=XP.shape)).ravel(), method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15)
            if np.sqrt(np.mean(sol.fun ** 2)) > 1e-11: continue                  # local minimum, not a realisation
            X = sol.x.reshape(-1, 3); Xc, Pc = X - X.mean(0), XP - XP.mean(0); best = np.inf
            for refl in (1, -1):
                A = Xc * np.array([1, 1, refl]); Uu, S, Vt = np.linalg.svd(A.T @ Pc); dd = np.sign(np.linalg.det(Uu @ Vt))
                best = min(best, np.sqrt((((A @ (Uu @ np.diag([1, 1, dd]) @ Vt)) - Pc) ** 2).sum(1).mean()))
            devs.append(best)
        row['rebuild'] = dict(converged=len(devs), starts=40, max_rmsd=float(max(devs)) if devs else None)
    rows.append(row)
    print(f"level {lev}: V={len(XP):5d}  forward {U['residual']:.1e}  inverse {res:.1e}  edge-length error {err:.1e}"
          + (f"  | rebuild: {row['rebuild']['converged']}/40 exact, max RMSD {row['rebuild']['max_rmsd']:.1e}" if lev == 0 else ''), flush=True)
json.dump(rows, open(os.path.join(out, 'troyanov_roundtrip.json'), 'w'), indent=1)
