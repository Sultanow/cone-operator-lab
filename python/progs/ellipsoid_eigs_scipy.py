import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

# Run the program:
# c:/Users/sulta/AppData/Local/Programs/Python/Python310/python ellipsoid_eigs_scipy.py
def build_ellipsoid_laplacian(a=1.0, b=1.5, c=2.3,
                              nx=40, ny=40, nz=40):
    """
    Generate the discrete Laplacian matrix (Dirichlet) on an
    Ellipsoid x^2/a^2 + y^2/b^2 + z^2/c^2 <= 1 with
    finite differences on a Cartesian grid.

    Return:
        L : sparse.csr_matrix   (N x N)
        points : ndarray (N, 3)  Gitterpunkte (x,y,z) der inneren Freiheitsgrade
    """

    # Grid in Bounding Box [-a,a] x [-b,b] x [-c,c]
    x = np.linspace(-a, a, nx)
    y = np.linspace(-b, b, ny)
    z = np.linspace(-c, c, nz)

    hx = x[1] - x[0]
    hy = y[1] - y[0]
    hz = z[1] - z[0]

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Points inside the ellipsoid
    inside = (X / a) ** 2 + (Y / b) ** 2 + (Z / c) ** 2 <= 1.0

    # Mapping (i,j,k) to running Index 0..N-1 for inner points
    idx_map = -np.ones_like(inside, dtype=int)
    idx_map[inside] = np.arange(inside.sum())

    n_unknowns = inside.sum()
    print(f"Number of degrees of freedom in the ellipsoid: {n_unknowns}")

    rows = []
    cols = []
    data = []

    # Coefficients for 3D-Laplacian (standard 7-Point-Stencil):
    c_x = 1.0 / hx ** 2
    c_y = 1.0 / hy ** 2
    c_z = 1.0 / hz ** 2

    # -Δu ≈ (2/hx^2 + 2/hy^2 + 2/hz^2) u_i
    #       - 1/hx^2 (u_{i+1} + u_{i-1})
    #       - 1/hy^2 (u_{j+1} + u_{j-1})
    #       - 1/hz^2 (u_{k+1} + u_{k-1})
    nx_, ny_, nz_ = inside.shape

    for i in range(nx_):
        for j in range(ny_):
            for k in range(nz_):
                if not inside[i, j, k]:
                    continue

                p = idx_map[i, j, k]

                diag = 2.0 * (c_x + c_y + c_z)

                # x direction
                for di in (-1, 1):
                    ii = i + di
                    if 0 <= ii < nx_:
                        if inside[ii, j, k]:
                            q = idx_map[ii, j, k]
                            rows.append(p)
                            cols.append(q)
                            data.append(-c_x)
                        else:
                            # Neighbor is outside -> Dirichlet-0, contributes
                            # only inhomogeneously (here homogeneous, i.e., 0)
                            pass

                # y direction
                for dj in (-1, 1):
                    jj = j + dj
                    if 0 <= jj < ny_:
                        if inside[i, jj, k]:
                            q = idx_map[i, jj, k]
                            rows.append(p)
                            cols.append(q)
                            data.append(-c_y)

                # z direction
                for dk in (-1, 1):
                    kk = k + dk
                    if 0 <= kk < nz_:
                        if inside[i, j, kk]:
                            q = idx_map[i, j, kk]
                            rows.append(p)
                            cols.append(q)
                            data.append(-c_z)

                # Diagonal
                rows.append(p)
                cols.append(p)
                data.append(diag)

    L = sparse.csr_matrix((data, (rows, cols)), shape=(n_unknowns, n_unknowns))

    # Coordinates of the inner points (optional, if we want to plot modes)
    inner_points = np.vstack([X[inside], Y[inside], Z[inside]]).T

    return L, inner_points


def compute_dirichlet_eigenvalues(a=1.0, b=1.5, c=2.3,
                                  nx=40, ny=40, nz=40,
                                  num_eigs=50,
                                  tol=1e-8,
                                  maxiter=50000):
    """
    Build -Δ on the Ellipsoid and calculate the first (num_eigs)
    Dirichlet eigenvalues using scipy.sparse.linalg.eigsh (Arnoldi/ARPACK).
    """

    L, points = build_ellipsoid_laplacian(a, b, c, nx, ny, nz)

    # Security check: num_eigs < dimension
    n = L.shape[0]
    if num_eigs >= n:
        raise ValueError(f"num_eigs={num_eigs} >= Dimension={n}. "
                         f"Reduziere num_eigs oder erhöhe die Gitterauflösung.")

    print("Starting Arnoldi/ARPACK (eigsh)...")
    # Smallest eigenvalues of SPD matrix: which=‘SM’ (smallest magnitude)
    vals, vecs = eigsh(L,
                       k=num_eigs,
                       which='SM',
                       tol=tol,
                       maxiter=maxiter)

    # sorting
    idx = np.argsort(vals)
    vals_sorted = vals[idx]
    vecs_sorted = vecs[:, idx]

    return vals_sorted, vecs_sorted, points


if __name__ == "__main__":
    a = 1.0
    b = 1.5
    c = 2.3

    # Resolution of the mesh
    # For 1000 eigenvalues use nx=ny=nz=60 (or finer)
    nx, ny, nz = 40, 40, 40

    num_eigs = 20

    eigenvalues, eigenvectors, points = compute_dirichlet_eigenvalues(
        a=a, b=b, c=c,
        nx=nx, ny=ny, nz=nz,
        num_eigs=num_eigs,
        tol=1e-8,
        maxiter=50000
    )

    print("\n--- First Dirichlet eigenvalues (3D-Volume, Ellipsoid) ---")
    for i, lam in enumerate(eigenvalues[:20], start=1):
        print(f"{i:3d}: {lam:.8f}")

    # Export to file
    np.savetxt(
        "ellipsoid_eigs_a1_b1.5_c2.3-20.txt",
        eigenvalues,
        fmt="%.12e"
    )
