from __future__ import annotations

import numpy as np


def true_ellipsoid_volume(a: float, b: float, c: float) -> float:
    return float(4.0 * np.pi * a * b * c / 3.0)


def ellipsoid_surface_area(a: float, b: float, c: float) -> float:
    """Surface area. Uses exact elliptic-integral formula when SciPy is available.

    Falls back to Knud Thomsen approximation.
    """
    p = 1.6075
    s_thomsen = 4 * np.pi * (((a*b)**p + (a*c)**p + (b*c)**p) / 3.0)**(1.0/p)

    try:
        from scipy.special import ellipeinc, ellipkinc
        A, B, C = sorted([float(a), float(b), float(c)], reverse=True)
        if abs(A - C) < 1e-14:
            return float(4 * np.pi * A**2)

        phi = np.arccos(C / A)
        k2 = A**2 * (B**2 - C**2) / (B**2 * (A**2 - C**2))
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        E = ellipeinc(phi, k2)
        F = ellipkinc(phi, k2)
        return float(
            2*np.pi*C**2
            + (2*np.pi*A*B/sin_phi)
            * (E*sin_phi**2 + F*cos_phi**2)
        )
    except Exception:
        return float(s_thomsen)

def ellipsoid_surface_integrands(
    theta: np.ndarray,
    phi: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(dS, Hbar, Hbar*dS)`` for the standard ellipsoid parametrization.

    The surface is parametrized by

        r(theta, phi) = (a sin(theta) cos(phi),
                         b sin(theta) sin(phi),
                         c cos(theta)).

    We use the convention ``Hbar = (kappa_1 + kappa_2)/2`` and choose the sign so
    that the outward mean curvature of a sphere is positive. This matches the
    convention used for the Weyl curvature coefficient in the paper.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)

    # First derivatives.
    r_t = np.stack([a * ct * cp, b * ct * sp, -c * st], axis=0)
    r_p = np.stack([-a * st * sp, b * st * cp, np.zeros_like(theta + phi)], axis=0)

    # Second derivatives.
    r_tt = np.stack([-a * st * cp, -b * st * sp, -c * ct], axis=0)
    r_tp = np.stack([-a * ct * sp, b * ct * cp, np.zeros_like(theta + phi)], axis=0)
    r_pp = np.stack([-a * st * cp, -b * st * sp, np.zeros_like(theta + phi)], axis=0)

    E1 = np.sum(r_t * r_t, axis=0)
    F1 = np.sum(r_t * r_p, axis=0)
    G1 = np.sum(r_p * r_p, axis=0)

    n_vec = np.cross(np.moveaxis(r_t, 0, -1), np.moveaxis(r_p, 0, -1))
    dS = np.linalg.norm(n_vec, axis=-1)
    n_hat = np.moveaxis(n_vec / dS[..., None], -1, 0)

    # Second fundamental form. With this parametrization and outward normal, the
    # sign of the textbook expression is negative on a sphere, hence the minus.
    e2 = np.sum(n_hat * r_tt, axis=0)
    f2 = np.sum(n_hat * r_tp, axis=0)
    g2 = np.sum(n_hat * r_pp, axis=0)
    denom = 2.0 * (E1 * G1 - F1**2)
    Hbar = -(e2 * G1 - 2.0 * f2 * F1 + g2 * E1) / denom

    return dS, Hbar, Hbar * dS


def ellipsoid_integrated_mean_curvature(
    a: float,
    b: float,
    c: float,
    *,
    n_theta: int = 240,
    n_phi: int = 480,
) -> float:
    """Numerically integrate the mean curvature over a triaxial ellipsoid.

    The value returned is

        C(a,b,c) = integral_{partial Omega(a,b,c)} Hbar dS,

    where ``Hbar = (kappa_1 + kappa_2)/2``.  For a sphere of radius ``R`` this
    gives ``4*pi*R``.
    """
    if min(a, b, c) <= 0:
        raise ValueError("semi-axes must be positive")
    if n_theta < 8 or n_phi < 16:
        raise ValueError("use at least n_theta=8 and n_phi=16")

    # Gauss-Legendre on theta in [0, pi]; trapezoidal rule in periodic phi.
    x, w = np.polynomial.legendre.leggauss(n_theta)
    theta = 0.5 * np.pi * (x + 1.0)
    w_theta = 0.5 * np.pi * w
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi

    th, ph = np.meshgrid(theta, phi, indexing="ij")
    _, _, integrand = ellipsoid_surface_integrands(th, ph, a, b, c)
    return float(np.sum(integrand * w_theta[:, None]) * dphi)
