# laplace_ellipsoid.py
import numpy as np
import scipy.sparse as sp

def build_laplace_ellipsoid(a=1.0, b=1.5, c=2.3, nx=31, ny=31, nz=31):
    """
    Dirichlet-Laplace-Operator (-Δ) on the Ellipsoid
    x^2/a^2 + y^2/b^2 + z^2/c^2 <= 1
    Discretization: 3D FD, 2. order
    Return: scipy.sparse.csr_matrix
    """

    x = np.linspace(-a, a, nx)
    y = np.linspace(-b, b, ny)
    z = np.linspace(-c, c, nz)

    hx = x[1] - x[0]
    hy = y[1] - y[0]
    hz = z[1] - z[0]

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    inside = (X/a)**2 + (Y/b)**2 + (Z/c)**2 <= 1.0

    idx = -np.ones_like(X, dtype=int)
    counter = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if inside[i, j, k]:
                    idx[i, j, k] = counter
                    counter += 1

    N = counter

    rows, cols, data = [], [], []

    ix2 = 1.0 / hx**2
    iy2 = 1.0 / hy**2
    iz2 = 1.0 / hz**2
    diag = 2.0 * (ix2 + iy2 + iz2)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                p = idx[i, j, k]
                if p < 0:
                    continue

                rows.append(p)
                cols.append(p)
                data.append(diag)

                for ni, nj, nk, val in [
                    (i-1,j,k,ix2), (i+1,j,k,ix2),
                    (i,j-1,k,iy2), (i,j+1,k,iy2),
                    (i,j,k-1,iz2), (i,j,k+1,iz2),
                ]:
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        q = idx[ni, nj, nk]
                        if q >= 0:
                            rows.append(p)
                            cols.append(q)
                            data.append(-val)

    L_sparse = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return L_sparse
