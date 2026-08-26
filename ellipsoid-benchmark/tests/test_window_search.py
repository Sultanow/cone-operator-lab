import numpy as np

from hearing_ellipsoid_bench.validation.window_search import (
    StableWindowConfig,
    _window_relative_errors,
    search_stable_weyl_windows,
)


def _direct_errors(eigs, start, end, references):
    lam = eigs[start - 1 : end]
    counts = np.arange(start, end + 1, dtype=float)
    X = np.column_stack((lam**1.5, lam, lam**0.5))
    scale = np.linalg.norm(X, axis=0)
    coef = np.linalg.lstsq(X / scale, counts, rcond=None)[0] / scale
    estimates = np.array(
        (6 * np.pi**2 * coef[0], -16 * np.pi * coef[1], 6 * np.pi**2 * coef[2])
    )
    rel = (estimates - references) / references
    return rel, np.sqrt(np.mean(rel**2))


def test_prefix_fit_matches_direct_lstsq():
    eigs = np.linspace(2.0, 80.0, 40) ** 1.1
    references = np.array((14.0, 31.0, 21.0))
    starts = np.array((1, 3, 11), dtype=np.int32)
    ends = np.array((20, 31, 40), dtype=np.int32)

    v_err, s_err, c_err, geometry_err = _window_relative_errors(
        eigs, starts, ends, references, chunk_size=2
    )
    for i, (start, end) in enumerate(zip(starts, ends)):
        rel, rms = _direct_errors(eigs, int(start), int(end), references)
        np.testing.assert_allclose((v_err[i], s_err[i], c_err[i]), rel, rtol=1e-8)
        np.testing.assert_allclose(geometry_err[i], rms, rtol=1e-8)


def test_stable_search_uses_full_neighborhood_and_all_methods():
    base = np.linspace(3.0, 40.0, 14)
    spectra = {
        "one": base,
        "two": base * (1.0 + 1e-5),
    }
    config = StableWindowConfig(
        min_window=4,
        start_radius=1,
        end_radius=2,
        beta=1.0,
        chunk_size=10,
        top_k=5,
    )
    result = search_stable_weyl_windows(
        spectra,
        references=(14.0, 31.0, 21.0),
        config=config,
        common_spectral_cutoff=False,
    )

    assert result.meta["raw_windows_evaluated_per_method"] == 66
    assert result.meta["methods"] == ["one", "two"]
    assert len(result.top_geometry) == 5
    assert len(result.top_a2) == 5
    assert set(result.best_geometry_by_method["method"]) == {"one", "two"}
    assert set(result.best_a2_by_method["method"]) == {"one", "two"}
    assert set(result.errors_at_geometry_best["method"]) == {"one", "two"}

    best = result.best_geometry
    assert best["fit_start"] - config.start_radius >= 1
    assert best["fit_end"] + config.end_radius <= result.meta["common_index_count"]
    assert (
        best["fit_end"]
        - config.end_radius
        - (best["fit_start"] + config.start_radius)
        + 1
        >= config.min_window
    )
