from laplace_ellipsoid import build_laplace_ellipsoid
from scipy.sparse.linalg import eigsh

L = build_laplace_ellipsoid(nx=31, ny=31, nz=31)

evals = eigsh(L, k=100, sigma=0.0, which="LM",
              tol=1e-10, maxiter=200000,
              return_eigenvectors=False)

print(evals[:20])
