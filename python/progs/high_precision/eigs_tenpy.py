# ------------------------------------------------------------
# Run:
# cd c:/Users/sulta/git/cone-operator-lab/python/progs/high_precision
# c:/Users/sulta/AppData/Local/Programs/Python/Python310/python eigs_tenpy.py
# ------------------------------------------------------------
import os
import numpy as np
from scipy.sparse.linalg import splu
from tenpy.linalg import np_conserved as npc
from tenpy.linalg.krylov_based import Arnoldi

from laplace_ellipsoid import build_laplace_ellipsoid


# ------------------------------------------------------------
# Parameter
# ------------------------------------------------------------
nx = ny = nz = 31
num_ev = 400  # gewünschte Anzahl (TenPy kann weniger zurückgeben)
out_dir = r"C:\Users\sulta\git\cone-operator-lab\data\eigenvalues"
os.makedirs(out_dir, exist_ok=True)

method_tag = f"tenpy_shiftinvert_grid{nx}_k{num_ev}"


# ------------------------------------------------------------
# Operator aufbauen
# ------------------------------------------------------------
L = build_laplace_ellipsoid(nx=nx, ny=ny, nz=nz)
N = L.shape[0]
print("DOFs:", N)

lu = splu(L.tocsc())


class ShiftInvertTenpyOperator:
    def __init__(self, lu):
        self.lu = lu
        self.dtype = float

    def matvec(self, psi):
        v = psi.to_ndarray().ravel()
        w = self.lu.solve(v)  # (L - σI)^(-1) mit σ = 0
        return npc.Array.from_ndarray_trivial(w, labels=["p"])


# ------------------------------------------------------------
# Arnoldi
# ------------------------------------------------------------
rng = np.random.default_rng(42)
psi0_np = rng.normal(size=N)
psi0_np /= np.linalg.norm(psi0_np)
psi0 = npc.Array.from_ndarray_trivial(psi0_np, labels=["p"])

# TenPy akzeptiert max_iter hier offenbar nicht -> weglassen
arn = Arnoldi(
    ShiftInvertTenpyOperator(lu),
    psi0,
    {
        "num_ev": num_ev,
        "which": "LM",
        "N_max": max(4 * num_ev, 200),
        "P_tol": 1e-12,
    }
)

mu, vecs, krylov_dim = arn.run()

# ------------------------------------------------------------
# Rücktransformation + Robust-Filtern
# ------------------------------------------------------------
mu = np.asarray(mu)  # kann complex sein
n_pairs = min(len(mu), len(vecs))  # schützt vor IndexError

mu_eps = 1e-14  # alles darunter ist praktisch 0 -> raus
pairs = []

for i in range(n_pairs):
    mui = mu[i]
    vi = vecs[i]

    # komplexe Ritzwerte: wenn Im-Teil klein, nimm real; sonst nimm Betrag
    if np.iscomplexobj(mui):
        if abs(mui.imag) <= 1e-10 * max(1.0, abs(mui.real)):
            mui = mui.real
        else:
            mui = abs(mui)

    # mu zu klein -> 1/mu explodiert
    if not np.isfinite(mui) or abs(mui) < mu_eps:
        continue

    lam = 1.0 / mui  # sigma = 0

    # Laplace-Dirichlet -> positive Eigenwerte
    if not np.isfinite(lam) or lam <= 0.0:
        continue

    pairs.append((lam, vi))

# sortieren nach λ
pairs.sort(key=lambda t: t[0])

if len(pairs) == 0:
    raise RuntimeError("No valid eigenpairs produced by TenPy Arnoldi. Try increasing N_max or relaxing tolerances.")

# ggf. auf num_ev begrenzen
pairs = pairs[:min(num_ev, len(pairs))]

lam = np.array([p[0] for p in pairs], dtype=float)
vecs = [p[1] for p in pairs]

print(f"Valid eigenpairs: {len(lam)} (requested {num_ev})")
print("Erste 20 Eigenwerte:")
print(lam[:20])


# ------------------------------------------------------------
# Export
# ------------------------------------------------------------
txt_path = os.path.join(out_dir, f"ellipsoid_{method_tag}.txt")
np.savetxt(txt_path, lam, fmt="%.18e")
print("Eigenwerte gespeichert:", txt_path)

# Eigenvektoren (optional)
npz_path = os.path.join(out_dir, f"ellipsoid_{method_tag}.npz")
evecs_np = np.column_stack([v.to_ndarray().ravel() for v in vecs])

np.savez(
    npz_path,
    evals=lam,
    evecs=evecs_np,
    grid=(nx, ny, nz),
    dofs=N,
    method="tenpy_shiftinvert",
    krylov_dim=int(krylov_dim),
)
print("Eigenwerte + Eigenvektoren gespeichert:", npz_path)
