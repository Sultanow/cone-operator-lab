"""
bmfem.py -- assembly and solvers for the Brooks-Makover cusp/compactification study.

* stiffness / mass:   K_ij = int grad phi_i . grad phi_j dxdy   (chart-Euclidean, exact
                      for the hyperbolic Laplacian by conformal invariance),
                      M_ij(w) = int w phi_i phi_j dxdy with vertex-interpolated weight w.
* cusped surface S:   L^2 eigenvalues below 1/4 via the exact cusp Dirichlet-to-Neumann
                      map on the truncation horocycles (Fourier modes m, K_nu Bessel),
                      solved as a nonlinear eigenproblem by fixed-point iteration.
* compactified S^bar: Liouville equation  Delta_0 phi = K_0 + e^{2 phi}  for the
                      hyperbolic metric  g = e^{2 phi} g_0  (Newton on the convex energy),
                      then  K u = lambda M(rho_0 e^{2 phi}) u.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import kve

from bmsurf import Mesh

FOUR_PI = 4 * math.pi


# ----------------------------------------------------------------------------
# assembly
# ----------------------------------------------------------------------------
def element_areas(xy: np.ndarray) -> np.ndarray:
    x, y = xy[:, :, 0], xy[:, :, 1]
    a2 = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    if not (a2 > 0).all():
        raise RuntimeError("%d elements with non-positive area" % int((a2 <= 0).sum()))
    return 0.5 * a2


def _assemble(tri, Ke, nn):
    rows = np.repeat(tri[:, :, None], 3, axis=2).ravel()
    cols = np.repeat(tri[:, None, :], 3, axis=1).ravel()
    return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(nn, nn)).tocsr()


def stiffness(tri, xy, nn):
    x, y = xy[:, :, 0], xy[:, :, 1]
    A = element_areas(xy)
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1)
    Ke = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (4 * A)[:, None, None]
    return _assemble(tri, Ke, nn)


# Dunavant degree-4 rule (6 points), barycentric coordinates and weights (sum 1)
_a, _b = 0.816847572980459, 0.091576213509771
_c, _d = 0.108103018168070, 0.445948490915965
_QP = np.array([[_a, _b, _b], [_b, _a, _b], [_b, _b, _a], [_c, _d, _d], [_d, _c, _d], [_d, _d, _c]])
_QW = np.array([0.109951743655322] * 3 + [0.223381589678011] * 3)


def mass(tri, xy, kind, rho_const, nn, phi=None):
    """Consistent mass matrix  M_ij = int rho_0 e^{2 phi} phi_i phi_j dxdy  with the exact
    chart density rho_0 (1/y^2 or constant) evaluated at the quadrature points and phi P1."""
    A = element_areas(xy)
    Me = np.zeros((len(tri), 3, 3))
    y_v = xy[:, :, 1]
    ph_v = None if phi is None else phi[tri]
    for q, wq in zip(_QP, _QW):
        yq = np.where(kind == 0, y_v @ q, 1.0)
        rq = np.where(kind == 0, 1.0 / (yq * yq), rho_const)
        if ph_v is not None:
            rq = rq * np.exp(2.0 * (ph_v @ q))
        Me += (wq * A * rq)[:, None, None] * np.outer(q, q)[None, :, :]
    return _assemble(tri, Me, nn)


def lumped(M):
    return np.asarray(M.sum(axis=1)).ravel()


def restrict(A, idx):
    return A.tocsr()[idx][:, idx].tocsc()


# ----------------------------------------------------------------------------
# cusp horocycle integrals
# ----------------------------------------------------------------------------
_GL_T, _GL_W = np.polynomial.legendre.leggauss(4)
_GL_T = 0.5 * (_GL_T + 1)
_GL_W = 0.5 * _GL_W


def chain_segments(ch):
    """Cyclic segment lengths and hat integrals (int phi_i dx_c) on a cusp chain."""
    x, k = ch.x, ch.k
    dx = np.diff(np.append(x, x[0] + k))
    hat = 0.5 * (dx + np.roll(dx, 1))
    return dx, hat


def chain_fourier(ch, M):
    """B[i, m] = int phi_i(x) exp(2 pi i m x / k) dx  for m = -M..M  (4-pt Gauss)."""
    x, k, N = ch.x, ch.k, len(ch.x)
    dx, _ = chain_segments(ch)
    ms = np.arange(-M, M + 1)
    B = np.zeros((N, len(ms)), dtype=complex)
    for t, w in zip(_GL_T, _GL_W):
        xg = x + t * dx
        e = np.exp(2j * math.pi * np.outer(xg, ms) / k) * (w * dx)[:, None]
        B += e * (1 - t)                       # hat of node i on segment i
        B[(np.arange(N) + 1) % N] += e * t     # hat of node i+1 on segment i
    return ms, B


def dtn_logderivs(ms, nu, Y, k):
    """d/dy log of the L^2 cusp mode  y^{1/2-nu}  (m=0)  resp.  sqrt(y) K_nu(2 pi |m| y/k)."""
    ell = np.empty(len(ms))
    for i, m in enumerate(ms):
        if m == 0:
            ell[i] = (0.5 - nu) / Y
        else:
            a = 2 * math.pi * abs(m) / k
            z = a * Y
            ell[i] = 0.5 / Y + a * (-kve(nu - 1, z) / kve(nu, z) - nu / z)
    return ell


class CuspDtN:
    """Assembled DtN operator  D(nu) = sum_c sum_m (ell_m / k) Re(B_m B_m^H)  on given node indexing."""

    def __init__(self, chains, loc, nn):
        self.parts = []
        for ch in chains:
            M = len(ch.x) // 2
            ms, B = chain_fourier(ch, M)
            self.parts.append((ch, ms, B, loc[ch.node]))
        self.nn = nn
        # constant-mode check needs the boundary "mass": (b^T 1)= k
        for ch, ms, B, idx in self.parts:
            b0 = B[:, list(ms).index(0)].real
            assert abs(b0.sum() - ch.k) < 1e-8 * ch.k

    def matrix(self, nu):
        rows, cols, vals = [], [], []
        for ch, ms, B, idx in self.parts:
            ell = dtn_logderivs(ms, nu, ch.Y, ch.k)
            D = (B * (ell / ch.k)[None, :]) @ B.conj().T
            D = D.real
            I, Jx = np.meshgrid(idx, idx, indexing="ij")
            rows.append(I.ravel()); cols.append(Jx.ravel()); vals.append(D.ravel())
        return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(self.nn, self.nn)).tocsc()


# ----------------------------------------------------------------------------
# eigen-solvers
# ----------------------------------------------------------------------------
def lowest(A, M, k, sigma=-0.02, v0=None):
    vals, vecs = spla.eigsh(A.tocsc(), k=k, M=M.tocsc(), sigma=sigma, which="LM", v0=v0)
    o = np.argsort(vals)
    return vals[o], vecs[:, o]


def _is_constant_like(u, M, ones_M, area, tol=0.5):
    """True modes of the frozen DtN operator are (nearly) M-orthogonal to constants
    (variance fraction ~ 1); the spurious near-zero mode has variance fraction << 1."""
    uMu = u @ (M @ u)
    mean = (ones_M @ u) / area
    var = uMu - area * mean * mean
    return var / uMu < tol


def neumann_eigs(K, M, neig, sigma=-0.02):
    return lowest(K, M, neig + 1, sigma)


def cusp_eigs_dtn(K, M, dtn: CuspDtN, lam_start=None, tol=1e-10, log=None):
    """Smallest L^2 eigenvalue of the cusped surface below 1/4 (None if there is none).

    The exact cusp condition is the nonlinear boundary term D(nu), nu = sqrt(1/4 - lam)
    (y^{1/2-nu} for the zero mode, sqrt(y) K_nu for the others).  With
    g(lam) = lambda_1(K - D(nu(lam)), M) - lam,  g is decreasing (D grows with lam), so the
    L^2 eigenvalue is the unique root of g on (0, 1/4), which exists iff g(1/4^-) < 0.
    The spurious constant-like mode of the frozen operator is filtered out.
    Everything here is a converging FEM approximation (P1, O(h^2)); the Fourier truncation
    at the DtN boundary Y'' is not the limiting error (non-zero modes are ~exp(-2 pi Y''/k)
    there).  A 'None' result means: no discrete eigenvalue below 1/4 was found -- it is a
    numerical statement, not a certificate (that would need validated numerics).
    """
    ones = np.ones(K.shape[0])
    ones_M = M @ ones
    area = ones_M @ ones
    cache = {}

    def frozen(nu):
        A = K - dtn.matrix(nu)
        vals, vecs = lowest(A, M, 4, sigma=-0.02)
        for i in range(len(vals)):
            if not _is_constant_like(vecs[:, i], M, ones_M, area):
                return vals[i]
        raise RuntimeError("only constant-like modes found")

    def g(lam):
        nu = math.sqrt(max(0.25 - lam, 0.0))
        val = frozen(nu)
        cache[lam] = val
        if log:
            log("  DtN: lam=%.10f  nu=%.6f  lambda_1(frozen)=%.10f" % (lam, nu, val))
        return val - lam

    hi = 0.25 - 1e-9
    g_hi = g(hi)
    if g_hi >= 0:
        return None, dict(note="no L2 eigenvalue below 1/4 detected in the discretisation "
                               "(frozen nu=0 value %.6f); numerical statement, not a certificate" % cache[hi],
                          frozen_nu0=float(cache[hi]), evaluations=len(cache), certified=False,
                          converged=True, status="no_l2_detected")
    lo = 1e-6
    g_lo = g(lo)
    if g_lo <= 0:
        return None, dict(note="frozen eigenvalue non-positive at lam->0 (surface disconnected?)",
                          certified=False, converged=False, status="invalid_lower_endpoint")
    # Illinois (modified regula falsi) on the bracket, reusing the endpoint evaluations
    a, fa, b, fb, side = lo, g_lo, hi, g_hi, 0
    for _ in range(60):
        x = (a * fb - b * fa) / (fb - fa)
        fx = g(x)
        if abs(fx) < tol or abs(b - a) < tol:
            return float(x), dict(evaluations=len(cache), frozen_nu0=float(cache[hi]),
                                  certified=False, converged=True, status="root_converged")
        if fx * fb < 0:
            a, fa, side_new = b, fb, -1
        else:
            side_new = 1
            if side == 1:
                fa *= 0.5
        b, fb, side = x, fx, side_new
        if fx * fa > 0 and side_new == 1:
            pass
    return float(x), dict(evaluations=len(cache), frozen_nu0=float(cache[hi]),
                          certified=False, converged=False, status="root_not_converged",
                          note="root search not converged")


# ----------------------------------------------------------------------------
# Liouville uniformization of the compactification
# ----------------------------------------------------------------------------
def curvature_source(mesh: Mesh, M0_sy_lumped, nn):
    """c_i = int phi_i dK_0:  -dA_hyp on S_Y, + (2 pi / L_c - 1) ds on each truncation
    horocycle (jump of geodesic curvature between cusp metric and flat cap)."""
    c = -M0_sy_lumped.copy()
    for ch in mesh.cap_chains:
        _, hat = chain_segments(ch)
        c[ch.node] += (2 * math.pi / ch.L - 1.0) * hat / ch.Y      # ds_hyp = dx_c / Y
    return c


def solve_liouville(K, c, ML, tol=1e-11, maxit=60, log=None):
    """Newton with backtracking on E(phi) = 1/2 phi^T K phi + c^T phi + 1/2 sum ML e^{2 phi}."""
    nn = K.shape[0]
    phi = np.zeros(nn)

    def energy(p):
        return 0.5 * p @ (K @ p) + c @ p + 0.5 * (ML * np.exp(2 * p)).sum()

    E = energy(phi)
    for it in range(maxit):
        e2 = np.exp(2 * phi)
        R = K @ phi + c + ML * e2
        rn = np.abs(R).max()
        if log:
            log("  Liouville it %2d: |R|_inf=%.3e  E=%.12f" % (it, rn, E))
        if rn < tol:
            return phi, dict(iterations=it, residual=rn, converged=True, status="converged", tol=tol, maxit=maxit)
        H = (K + sp.diags(2 * ML * e2)).tocsc()
        dphi = -spla.splu(H).solve(R)
        t, slope = 1.0, R @ dphi
        while True:
            En = energy(phi + t * dphi)
            if En <= E + 1e-4 * t * slope or t < 1e-6:
                break
            t *= 0.5
        phi, E = phi + t * dphi, En
    return phi, dict(iterations=maxit, residual=float(np.abs(K @ phi + c + ML * np.exp(2 * phi)).max()),
                     converged=False, status="max_iterations", tol=tol, maxit=maxit,
                     note="Newton not converged")


def weyl_fit(vals, area_expected=None, use_fraction=0.6):
    """Least-squares slope of the counting function N(lam) = j against lam_j;
    Weyl: slope -> Area/(4 pi) = g - 1 for a closed hyperbolic surface."""
    lam = np.asarray(vals)
    lam = lam[lam > 1e-9]
    m = max(5, int(len(lam) * use_fraction))
    lam = lam[:m]
    j = np.arange(1, m + 1)
    slope, icpt = np.polyfit(lam, j, 1)
    out = dict(slope=float(slope), intercept=float(icpt), heard_genus=float(slope + 1), n_used=int(m))
    if area_expected is not None:
        out["slope_expected"] = area_expected / FOUR_PI
    return out
