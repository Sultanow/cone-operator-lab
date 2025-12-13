import math
import numpy as np
from sklearn.linear_model import Ridge

def fit_weyl_3_ridge(eigs, frac_range=(0.2, 0.9), alpha=1e-6):
    eigs = np.asarray(eigs, dtype=float)
    n = eigs.size

    r1, r2 = frac_range
    k_min = math.ceil(r1 * n)
    k_max = math.floor(r2 * n)

    lam = eigs[k_min-1 : k_max]
    k_vals = np.arange(k_min, k_max + 1, dtype=float)

    # Designmatrix: [λ^(3/2), λ, sqrt(λ)]
    X = np.column_stack([lam**1.5, lam, np.sqrt(lam)])

    # Ridge-Regression: (X^T X + alpha I)^{-1} X^T y
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, k_vals)
    # model.coef_ = [A0, A1, A2]
    return model.coef_

def volume_from_A0(A0):
    """V = 6 π^2 A0"""
    return 6.0 * np.pi**2 * float(A0)


def true_ellipsoid_volume(a, b, c):
    """V_true = 4π/3 abc"""
    return 4.0 * np.pi / 3.0 * float(a) * float(b) * float(c)


def volume_from_eigs(eigs, frac_range=(0.2, 0.9), alpha=1e-6):
    """Gibt (V_spec, A0, A1, A2) zurück."""
    A0, A1, A2 = fit_weyl_3_ridge(eigs, frac_range=frac_range, alpha=alpha)
    V_spec = volume_from_A0(A0)
    return float(V_spec), float(A0), float(A1), float(A2)

def volume_quality_from_eigs(eigs, a=1.0, b=1.5, c=2.3, frac_range=(0.2, 0.9), alpha=1e-6):
    A0, A1, A2 = fit_weyl_3_ridge(eigs, frac_range=frac_range, alpha=alpha)
    V_spec = volume_from_A0(A0)
    V_true = true_ellipsoid_volume(a, b, c)
    rel_vol_err = abs(V_spec - V_true) / V_true
    return V_spec, V_true, rel_vol_err, (A0, A1, A2)

def vol_from_first_m(eigs, frac_range, alpha):
    A0_sub, _, _ = fit_weyl_3_ridge(
        eigs,
        frac_range=frac_range,
        alpha=alpha
    )
    return 6.0 * np.pi**2 * A0_sub
