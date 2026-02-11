import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.linalg import lstsq
from dataclasses import dataclass

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import lstsq

PI = math.pi

# ----------------------------
# Geometry / exact coefficient
# ----------------------------
@dataclass(frozen=True)
class Box:
    a: float
    b: float
    c: float

    @property
    def A2_exact(self) -> float:
        # Eq. (9): A2 = (a+b+c)/(4*pi^2)
        return (self.a + self.b + self.c) / (4 * PI**2)


# ----------------------------
# Core building block (single source of truth)
# ----------------------------
def _heat_fit_design_and_Z(eigs: np.ndarray, t_min: float, t_max: float, m_grid: int):
    """
    Build log-spaced grid, truncated heat trace Z(t), and design matrix X.
    X columns: [t^-3/2, t^-1, t^-1/2, 1]
    """
    eigs = np.asarray(eigs, dtype=float)
    if eigs.ndim != 1 or eigs.size == 0:
        raise ValueError("eigs must be a non-empty 1D array")
    if t_min <= 0 or t_max <= 0 or t_max <= t_min:
        raise ValueError("Require 0 < t_min < t_max")
    if m_grid < 4:
        raise ValueError("m_grid must be >= 4")

    # reverse stays (matches your 'just for fun' test, harmless for LSQ)
    t = np.geomspace(t_min, t_max, m_grid)[::-1]
    Z = np.exp(-np.outer(eigs, t)).sum(axis=0)
    X = np.column_stack([t ** (-1.5), t ** (-1.0), t ** (-0.5), np.ones_like(t)])
    return t, X, Z


# ----------------------------
# Public API (used by notebooks)
# ----------------------------
def generate_box_eigenvalues(a: float, b: float, c: float, max_idx: int = 50, N_keep: int = 20000) -> np.ndarray:
    # Mirrors Listing 1: enumerate integer triples up to max_idx, compute eigenvalues, sort, keep first N_keep.
    l = np.arange(1, max_idx + 1, dtype=float)
    n = np.arange(1, max_idx + 1, dtype=float)
    m = np.arange(1, max_idx + 1, dtype=float)

    lam = (PI**2) * ((l[:, None, None] / a) ** 2 + (n[None, :, None] / b) ** 2 + (m[None, None, :] / c) ** 2)
    lam = np.sort(lam.ravel())
    return lam[:N_keep].astype(float)


def compute_A2_hat(eigs, f, *, tmax_factor=500.0, m_grid=151, diagnostics=False):
    """
    Compute A2_hat for a given eigenvalue list and f = t_min * lambda_N.
    If diagnostics=True, also return cond(X) and relative residual.
    """
    eigs = np.asarray(eigs, dtype=float)
    lamN = float(eigs[-1])
    t_min = float(f) / lamN
    t_max = float(tmax_factor) / lamN

    _, X, Z = _heat_fit_design_and_Z(eigs, t_min, t_max, int(m_grid))

    coef, *_ = lstsq(X, Z)  # LAPACK-backed
    A2_hat = (coef[2] * (4 * PI) ** 1.5) / (4 * PI**3)

    if not diagnostics:
        return float(A2_hat)

    condX = float(np.linalg.cond(X))
    rel_res = float(np.linalg.norm(X @ coef - Z) / max(np.linalg.norm(Z), 1e-300))
    return float(A2_hat), condX, rel_res


def scan_A2_grid(eigs_all, n_steps, factors, *, tmax_factor=500.0, m_grid=151, diagnostics=False):
    """
    Compute results for all (N,f) once and return a DataFrame.
    Columns:
      - always: N, f, A2_hat
      - if diagnostics: condX, rel_residual
    Robustness:
      - skips invalid (N,f) where f<=0 or f>=tmax_factor (=> t_min >= t_max)
      - skips out-of-range N
      - skips numerical failures
    """
    eigs_all = np.asarray(eigs_all, dtype=float).ravel()
    if eigs_all.size == 0:
        raise ValueError("eigs_all is empty")
    if not np.isfinite(tmax_factor) or tmax_factor <= 0:
        raise ValueError("tmax_factor must be finite and > 0")
    if m_grid < 4:
        raise ValueError("m_grid must be >= 4")

    factors = np.asarray(list(factors), dtype=float)
    # Keep only valid f: ensures 0 < t_min < t_max because t_min=f/lamN and t_max=tmax_factor/lamN
    factors = factors[np.isfinite(factors) & (factors > 0) & (factors < float(tmax_factor))]
    if factors.size == 0:
        raise ValueError("No valid factors left after filtering. Need 0 < f < tmax_factor.")

    rows = []
    n_max = eigs_all.size

    for N in n_steps:
        N = int(N)
        if N <= 0 or N > n_max:
            continue

        eigs = eigs_all[:N]

        for f in factors:
            try:
                out = compute_A2_hat(
                    eigs, float(f),
                    tmax_factor=float(tmax_factor),
                    m_grid=int(m_grid),
                    diagnostics=diagnostics
                )
            except (ValueError, FloatingPointError, OverflowError):
                # robust: skip bad (N,f) combos instead of aborting the whole run
                continue

            if diagnostics:
                A2_hat, condX, rel_res = out
                if not np.isfinite(A2_hat):
                    continue
                rows.append((N, float(f), float(A2_hat), float(condX), float(rel_res)))
            else:
                A2_hat = out
                if not np.isfinite(A2_hat):
                    continue
                rows.append((N, float(f), float(A2_hat)))

    if diagnostics:
        return pd.DataFrame(rows, columns=["N", "f", "A2_hat", "condX", "rel_residual"])
    return pd.DataFrame(rows, columns=["N", "f", "A2_hat"])


# ----------------------------
# Error + stability helpers (DF-based, reusable)
# ----------------------------

# calculates the absolute relative error |A2_hat / A2_exact - 1|
def add_A2_error(df: pd.DataFrame, A2_exact: float) -> pd.DataFrame:
    df = df.copy()
    df["rel_err"] = (df["A2_hat"] / float(A2_exact) - 1.0).abs()
    return df


def stability_minimum_from_grid(df: pd.DataFrame, A2_exact: float) -> pd.DataFrame:
    """
    Returns a df with one row per N:
    N, f_star, A2_hat, rel_err, min_rel_err_pct
    """
    df2 = add_A2_error(df, A2_exact)
    idx = df2.groupby("N")["rel_err"].idxmin()
    out = df2.loc[idx, ["N", "f", "A2_hat", "rel_err"]].rename(columns={"f": "f_star"})
    out = out.sort_values("N").reset_index(drop=True)
    out["min_rel_err_pct"] = 100 * out["rel_err"]
    return out


def scan_stability_minimum(
    eigs: np.ndarray,
    A2_exact: float,
    f_values=np.arange(1, 201),
    tmax_factor: float = 500.0,
    m_grid: int = 151,
    diagnostics: bool = False,
):
    """
    Tuple-API (backwards compatible):
      returns (f_vals, A2_hat, rel_err, f_star) [+ condX, rel_res if diagnostics]
    """
    df = scan_A2_grid(
        eigs_all=np.asarray(eigs, dtype=float),
        n_steps=[len(eigs)],
        factors=np.asarray(f_values, dtype=float),
        tmax_factor=tmax_factor,
        m_grid=m_grid,
        diagnostics=diagnostics,
    )
    df2 = add_A2_error(df, A2_exact)

    i = int(df2["rel_err"].to_numpy().argmin())
    f_star = float(df2.iloc[i]["f"])

    f_vals = df2["f"].to_numpy(dtype=float)
    A2_hat = df2["A2_hat"].to_numpy(dtype=float)
    rel_err = df2["rel_err"].to_numpy(dtype=float)

    if not diagnostics:
        return f_vals, A2_hat, rel_err, f_star

    condX = df2["condX"].to_numpy(dtype=float)
    rel_res = df2["rel_residual"].to_numpy(dtype=float)
    return f_vals, A2_hat, rel_err, f_star, condX, rel_res


# ----------------------------
# Optional: full alpha fit + Weyl conversion (only if you need A0/A1)
# ----------------------------
def heat_trace_fit(eigs: np.ndarray, f_min: float, f_max: float, grid: int = 250):
    """
    Returns (t_grid, Z, alpha) where alpha solves:
      Z(t) ≈ alpha0 t^-3/2 + alpha1 t^-1 + alpha2 t^-1/2 + alpha3
    """
    eigs = np.asarray(eigs, dtype=float)
    lamN = float(eigs[-1])

    t_min = float(f_min) / lamN
    t_max = float(f_max) / lamN

    t_grid, X, Z = _heat_fit_design_and_Z(eigs, t_min, t_max, int(grid))
    alpha, *_ = lstsq(X, Z)
    return t_grid, Z, alpha


def alpha_to_weyl(alpha: np.ndarray) -> np.ndarray:
    """
    Convert heat coefficients alpha -> Weyl coefficients (A0, A1, A2)
    using Listing-3 style constants.
    """
    alpha = np.asarray(alpha, dtype=float)
    alpha0, alpha1, alpha2, _alpha3 = alpha

    A0 = (alpha0 * (4 * PI) ** 1.5) / (6 * PI**2)
    A1 = alpha1
    A2 = (alpha2 * (4 * PI) ** 1.5) / (4 * PI**3)
    return np.array([A0, A1, A2], dtype=float)
