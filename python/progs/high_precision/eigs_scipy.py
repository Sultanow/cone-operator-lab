import os
import numpy as np
from scipy.sparse.linalg import eigsh

from laplace_ellipsoid import build_laplace_ellipsoid


# ------------------------------------------------------------
# Parameter
# ------------------------------------------------------------
nx = ny = nz = 31
num_ev = 100
sigma = 0.0

out_dir = r"C:\Users\sulta\git\cone-operator-lab\data\eigenvalues"
os.makedirs(out_dir, exist_ok=True)

method_tag = f"scipy_shiftinvert_grid{nx}_k{num_ev}"


# ------------------------------------------------------------
# Operator
# ------------------------------------------------------------
L = build_laplace_ellipsoid(nx=nx, ny=ny, nz=nz)
N = L.shape[0]
print("DOFs:", N)


# ------------------------------------------------------------
# SciPy Shift-Invert
# ------------------------------------------------------------
evals, evecs = eigsh(
    L,
    k=num_ev,
    sigma=sigma,
    which="LM",
    tol=1e-12,
    maxiter=500000,
    return_eigenvectors=True
)

# sortieren
idx = np.argsort(evals)
evals = evals[idx]
evecs = evecs[:, idx]

print("Erste 20 Eigenwerte:")
print(evals[:20])


# ------------------------------------------------------------
# Export
# ------------------------------------------------------------
# 1) Eigenwerte (.txt)
txt_path = os.path.join(out_dir, f"ellipsoid_{method_tag}.txt")
np.savetxt(txt_path, evals, fmt="%.18e")
print("Eigenwerte gespeichert:", txt_path)

# 2) Eigenwerte + Eigenvektoren (.npz) für Residual-Checks
npz_path = os.path.join(out_dir, f"ellipsoid_{method_tag}.npz")
np.savez(
    npz_path,
    evals=evals,
    evecs=evecs,
    grid=(nx, ny, nz),
    dofs=N,
    method="scipy_shiftinvert",
)
print("Eigenwerte + Eigenvektoren gespeichert:", npz_path)
