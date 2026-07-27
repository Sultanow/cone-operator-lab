from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hearing_ellipsoid_bench.core.types import clean_eigenvalues
from hearing_ellipsoid_bench.geometry.ellipsoid import (
    ellipsoid_integrated_mean_curvature,
    ellipsoid_surface_area,
    true_ellipsoid_volume,
)


@dataclass(frozen=True)
class WeylFit3:
    """Three-term Weyl fit in eigenvalue notation.

    N(lambda) = A0 lambda^(3/2) + A1 lambda + A2 lambda^(1/2).
    """

    A0: float
    A1: float
    A2: float

    @property
    def volume(self) -> float:
        return float(6.0 * np.pi**2 * self.A0)

    @property
    def surface_area(self) -> float:
        return float(-16.0 * np.pi * self.A1)

    @property
    def integrated_mean_curvature(self) -> float:
        # Important: C = 6*pi^2*A2 for Hbar=(kappa_1+kappa_2)/2.
        return float(6.0 * np.pi**2 * self.A2)


def _scaled_lstsq(eigs: np.ndarray, powers: tuple[float, ...]) -> np.ndarray:
    """Least-squares fit with column scaling for numerical stability."""
    e = clean_eigenvalues(eigs)
    counts = np.arange(1, len(e) + 1, dtype=float)
    scale = float(np.max(e))
    x = e / scale
    X_scaled = np.column_stack([x**p for p in powers])
    beta_scaled, *_ = np.linalg.lstsq(X_scaled, counts, rcond=None)
    return np.array([beta_scaled[i] / (scale ** p) for i, p in enumerate(powers)])


def fit_weyl_3term(eigs) -> WeylFit3:
    """Fit the three-term Weyl model and return coefficients A0, A1, A2."""
    e = clean_eigenvalues(eigs)
    if len(e) < 3:
        return WeylFit3(np.nan, np.nan, np.nan)
    A0, A1, A2 = _scaled_lstsq(e, (1.5, 1.0, 0.5))
    return WeylFit3(float(A0), float(A1), float(A2))


def weyl_recover_volume_surface_curvature(eigs) -> tuple[float, float, float]:
    """Recover volume, surface area, and integrated mean curvature from eigenvalues.

    Fits

        N(lambda)=A0 lambda^(3/2)+A1 lambda+A2 lambda^(1/2)

    and returns

        V = 6*pi^2*A0,
        S = -16*pi*A1,
        C = 6*pi^2*A2.
    """
    fit = fit_weyl_3term(eigs)
    return fit.volume, fit.surface_area, fit.integrated_mean_curvature


def weyl_recover_volume_surface(eigs) -> tuple[float, float]:
    """Fit N(lambda)=A lambda^(3/2)+B lambda.

    V=6*pi^2*A, S=-16*pi*B.
    """
    e = clean_eigenvalues(eigs)
    if len(e) < 2:
        return np.nan, np.nan
    counts = np.arange(1, len(e) + 1, dtype=float)
    X = np.column_stack([e**1.5, e])
    coef, *_ = np.linalg.lstsq(X, counts, rcond=None)
    A, B = coef
    return float(6.0 * np.pi**2 * A), float(-16.0 * np.pi * B)


def weyl_recover_volume_with_surface(eigs, surface_known: float) -> float:
    e = clean_eigenvalues(eigs)
    if len(e) == 0:
        return np.nan
    counts = np.arange(1, len(e) + 1, dtype=float)
    rhs = counts + surface_known / (16.0 * np.pi) * e
    X = e**1.5
    A = (X @ rhs) / (X @ X)
    return float(6.0 * np.pi**2 * A)


def reverse_geometry_table(
    eigs_by_method: dict[str, np.ndarray],
    a: float,
    b: float,
    c: float,
    k_values=(50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000),
    *,
    n_theta: int = 240,
    n_phi: int = 480,
) -> pd.DataFrame:
    V_true = true_ellipsoid_volume(a, b, c)
    S_true = ellipsoid_surface_area(a, b, c)
    C_true = ellipsoid_integrated_mean_curvature(a, b, c, n_theta=n_theta, n_phi=n_phi)
    rows = []
    for method, eigs in eigs_by_method.items():
        e = clean_eigenvalues(eigs)
        for K in k_values:
            if K > len(e) or K < 3:
                continue
            V_est_2term, S_est_2term = weyl_recover_volume_surface(e[:K])
            V_Sknown = weyl_recover_volume_with_surface(e[:K], S_true)
            fit = fit_weyl_3term(e[:K])
            rows.append(
                {
                    "method": method,
                    "K": int(K),
                    "A0_3term": fit.A0,
                    "A1_3term": fit.A1,
                    "A2_3term": fit.A2,
                    "V_est": V_est_2term,
                    "S_est": S_est_2term,
                    "V_est_S_known": V_Sknown,
                    "V_est_3term": fit.volume,
                    "S_est_3term": fit.surface_area,
                    "C_est_3term": fit.integrated_mean_curvature,
                    "V_true": V_true,
                    "S_true": S_true,
                    "C_true": C_true,
                    "V_rel_err": (V_est_2term - V_true) / V_true,
                    "S_rel_err": (S_est_2term - S_true) / S_true,
                    "V_rel_err_S_known": (V_Sknown - V_true) / V_true,
                    "V_rel_err_3term": (fit.volume - V_true) / V_true,
                    "S_rel_err_3term": (fit.surface_area - S_true) / S_true,
                    "C_rel_err_3term": (fit.integrated_mean_curvature - C_true) / C_true,
                }
            )
    return pd.DataFrame(rows)
