"""
Validated hyperbolic Laplace-Beltrami eigensolver (proof-of-concept "brick").

We solve the Dirichlet eigenvalue problem  -Delta_g u = lambda u  on a GEODESIC
DISK of hyperbolic radius R in H^2, using two INDEPENDENT methods:

  (1) FEM in the Poincare disk model, exploiting 2D conformal invariance:
        - stiffness  S_ij = int grad phi_i . grad phi_j dA_euclid   (metric-free in 2D)
        - mass       M_ij = int phi_i phi_j rho^2 dA_euclid,  rho = 2/(1-|z|^2)
      A geodesic disk of hyperbolic radius R centered at 0 is a EUCLIDEAN disk
      of radius a = tanh(R/2).

  (2) A 1D radial ODE ground truth (rotationally symmetric m=0 mode) in
      Sturm-Liouville form   -(sinh r * R')' = lambda * sinh r * R,
      R'(0)=0, R(Rmax)=0.  This is an exact reference for lambda_1.

If (1) reproduces (2), the hyperbolic mass matrix + solver core are correct.
"""
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

# ----------------------------------------------------------------------
# (2) GROUND TRUTH: radial ODE eigenvalues of the geodesic disk (m=0)
# ----------------------------------------------------------------------
def radial_ground_truth(Rmax, N=200000, k=4):
    # grid on (0, Rmax], staggered; SL form -(sinh r R')' = lam sinh r R
    r = np.linspace(0.0, Rmax, N + 1)
    dr = r[1] - r[0]
    rmid = 0.5 * (r[:-1] + r[1:])          # cell edges midpoints
    sh_mid = np.sinh(rmid)                  # sinh at edges
    sh_node = np.sinh(r)                    # sinh at nodes
    # unknowns: interior nodes 1..N-1  (node N is Dirichlet R=0; node 0 Neumann)
    # FD of -(sinh r R')' on node i: (sh_mid[i-1]*(R_i - R_{i-1}) - sh_mid[i]*(R_{i+1}-R_i))/dr^2
    n = N  # nodes 0..N, unknown 0..N-1 (drop Dirichlet node N)
    from scipy.sparse import diags
    lo = np.zeros(n); di = np.zeros(n); up = np.zeros(n)
    for i in range(n):
        left = sh_mid[i-1] if i - 1 >= 0 else 0.0     # edge to the left
        right = sh_mid[i] if i < N else 0.0           # edge to the right
        di[i] = (left + right) / dr**2
        if i - 1 >= 0:
            lo[i] = -left / dr**2
        if i + 1 <= n - 1:
            up[i] = -right / dr**2
    # Neumann at r=0 (node 0): only right edge contributes -> already handled (left=0)
    A = diags([lo[1:], di, up[:-1]], [-1, 0, 1]).tocsr()
    w = sh_node[:n].copy()
    w[0] = 0.5 * sh_node[1]  # small-r weight regularization; negligible effect
    B = diags([np.maximum(w, 1e-30)], [0]).tocsr()
    vals, _ = eigsh(A, k=k, M=B, sigma=0.0, which='LM')
    return np.sort(vals.real)


# ----------------------------------------------------------------------
# (1) FEM in Poincare disk
# ----------------------------------------------------------------------
def make_disk_mesh(a, nr=60, refine=True):
    """Structured-ish disk mesh of Euclidean radius a via concentric rings."""
    pts = [(0.0, 0.0)]
    for i in range(1, nr + 1):
        rad = a * (i / nr)
        ncirc = max(6, int(round(2 * np.pi * i)))  # roughly uniform spacing
        for j in range(ncirc):
            th = 2 * np.pi * j / ncirc
            pts.append((rad * np.cos(th), rad * np.sin(th)))
    pts = np.array(pts)
    tri = Delaunay(pts)
    return pts, tri.simplices


def fem_hyperbolic_disk(Rmax, nr=60, k=4):
    a = np.tanh(Rmax / 2.0)
    pts, tris = make_disk_mesh(a, nr=nr)
    Np = len(pts)
    S = lil_matrix((Np, Np))
    M = lil_matrix((Np, Np))
    rho2 = lambda z2: (2.0 / (1.0 - z2))**2  # z2 = |z|^2

    for t in tris:
        idx = t
        p = pts[idx]                       # 3x2
        # area + P1 gradients (Euclidean)
        x1, y1 = p[0]; x2, y2 = p[1]; x3, y3 = p[2]
        detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = 0.5 * abs(detJ)
        if area < 1e-14:
            continue
        b = np.array([y2 - y3, y3 - y1, y1 - y2]) / detJ
        c = np.array([x3 - x2, x1 - x3, x2 - x1]) / detJ
        # stiffness (conformally invariant in 2D -> Euclidean)
        Ke = area * (np.outer(b, b) + np.outer(c, c))
        # mass with hyperbolic weight rho^2, evaluated with 3 vertex-quadrature pts
        # weighted consistent mass: integrate phi_i phi_j rho^2 over triangle.
        # use 3-point (vertices) rule for rho^2 * exact P1 mass template.
        # exact P1 mass = area/12 * [[2,1,1],[1,2,1],[1,1,2]]; scale by mean rho^2.
        z2v = np.sum(p**2, axis=1)
        rho2_avg = np.mean(rho2(z2v))
        Me = (area / 12.0) * np.array([[2., 1., 1.],
                                        [1., 2., 1.],
                                        [1., 1., 2.]]) * rho2_avg
        for a_ in range(3):
            for b_ in range(3):
                S[idx[a_], idx[b_]] += Ke[a_, b_]
                M[idx[a_], idx[b_]] += Me[a_, b_]

    S = csr_matrix(S); M = csr_matrix(M)
    # Dirichlet: boundary nodes = outermost ring (|z| ~ a)
    r = np.sqrt(np.sum(pts**2, axis=1))
    interior = np.where(r < a - 1e-9)[0]
    S = S[interior][:, interior]
    M = M[interior][:, interior]
    vals, _ = eigsh(S, k=k, M=M, sigma=0.0, which='LM')
    return np.sort(vals.real), Np


if __name__ == "__main__":
    for Rmax in [1.0, 2.0]:
        gt = radial_ground_truth(Rmax, N=40000, k=3)
        print(f"\n=== Geodesic disk, hyperbolic radius R = {Rmax} "
              f"(Euclidean disk radius a = tanh(R/2) = {np.tanh(Rmax/2):.4f}) ===")
        print(f"  ODE ground truth  lambda_1 (m=0) = {gt[0]:.6f}")
        for nr in [40, 70, 100]:
            fem, Np = fem_hyperbolic_disk(Rmax, nr=nr, k=3)
            err = abs(fem[0] - gt[0]) / gt[0] * 100
            print(f"  FEM nr={nr:3d} (Np={Np:5d})  lambda_1 = {fem[0]:.6f}   "
                  f"rel.err = {err:5.2f}%")
