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
num_ev = 100
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
psi0 = npc.Array.from_ndarray_trivial(
    rng.normal(size=N),
    labels=["p"]
)

arn = Arnoldi(
    ShiftInvertTenpyOperator(lu),
    psi0,
    {
        "num_ev": num_ev,
        "which": "LM",
        "N_max": max(4 * num_ev, 200),
        "P_tol": 1e-12,
        "max_iter": 5000,
    }
)

mu, vecs, krylov_dim = arn.run()

# Rücktransformation: μ = 1/λ  →  λ = 1/μ
mu = np.asarray(mu, dtype=float)
lam = 1.0 / mu

# sortieren
idx = np.argsort(lam)
lam = lam[idx]
vecs = [vecs[i] for i in idx]


print("Erste 20 Eigenwerte:")
print(lam[:20])


# ------------------------------------------------------------
# Export
# ------------------------------------------------------------
txt_path = os.path.join(out_dir, f"ellipsoid_{method_tag}.txt")
np.savetxt(txt_path, lam, fmt="%.18e")

print("Eigenwerte gespeichert:", txt_path)


# Optional: Eigenvektoren für Residual-Checks speichern
npz_path = os.path.join(out_dir, f"ellipsoid_{method_tag}.npz")

evecs_np = np.column_stack([v.to_ndarray().ravel() for v in vecs])

np.savez(
    npz_path,
    evals=lam,
    evecs=evecs_np,
    grid=(nx, ny, nz),
    dofs=N,
    method="tenpy_shiftinvert",
)

print("Eigenwerte + Eigenvektoren gespeichert:", npz_path)
