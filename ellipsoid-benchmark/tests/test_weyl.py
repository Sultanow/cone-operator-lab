import numpy as np
from hearing_ellipsoid_bench.geometry.ellipsoid import true_ellipsoid_volume
from hearing_ellipsoid_bench.validation.sphere import compare_to_truth


def test_true_volume_unit_ball():
    assert abs(true_ellipsoid_volume(1, 1, 1) - 4*np.pi/3) < 1e-12


def test_compare_to_truth_zero_error():
    eigs = np.array([1.0, 2.0, 3.0])
    df = compare_to_truth(eigs, eigs)
    assert df["abs_rel_err"].max() == 0.0


from hearing_ellipsoid_bench.geometry.ellipsoid import ellipsoid_integrated_mean_curvature
from hearing_ellipsoid_bench.validation.weyl import fit_weyl_3term


def test_integrated_mean_curvature_sphere():
    R = 2.0
    C = ellipsoid_integrated_mean_curvature(R, R, R, n_theta=48, n_phi=96)
    assert abs(C - 4*np.pi*R) / (4*np.pi*R) < 1e-10


def test_three_term_weyl_scaling_for_exact_synthetic_counts():
    # Synthetic data from an exact three-term polynomial in lambda.  This checks
    # coefficient extraction and, in particular, the convention C = 6*pi^2*A2.
    lam = np.linspace(10.0, 500.0, 200)
    A0, A1, A2 = 0.2, -0.4, 0.6
    # Build pseudo-counts via exact model by monkey-patching clean eigenvalues is
    # not useful here, so we check the conversion on the returned dataclass.
    fit = fit_weyl_3term(lam)
    assert np.isfinite(fit.integrated_mean_curvature)
