import numpy as np
from scipy.sparse.linalg import splu
from tenpy.linalg import np_conserved as npc
from tenpy.linalg.krylov_based import Arnoldi
from laplace_ellipsoid import build_laplace_ellipsoid

L = build_laplace_ellipsoid(nx=31, ny=31, nz=31)
N = L.shape[0]

lu = splu(L.tocsc())

class ShiftInvertTenpyOperator:
    def __init__(self, lu):
        self.lu = lu
        self.dtype = float

    def matvec(self, psi):
        v = psi.to_ndarray().ravel()
        w = self.lu.solve(v)
        return npc.Array.from_ndarray_trivial(w, labels=["p"])

psi0 = npc.Array.from_ndarray_trivial(
    np.random.randn(N), labels=["p"]
)

arn = Arnoldi(
    ShiftInvertTenpyOperator(lu),
    psi0,
    {"num_ev": 100, "which": "LM"}
)

mu, vecs, _ = arn.run()
lam = 1.0 / np.array(mu)
print(lam[:10])
