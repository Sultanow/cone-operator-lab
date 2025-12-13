from scipy.sparse.linalg import splu
from laplace_ellipsoid import build_laplace_ellipsoid

L = build_laplace_ellipsoid(nx=31, ny=31, nz=31)
lu = splu(L.tocsc())

print("Matrixgröße:", L.shape)
print("nnz:", L.nnz)
print("cond ~", lu.U.diagonal().max() / lu.U.diagonal().min())