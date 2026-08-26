"""Stable exhaustive fit-window selection for three-term Weyl regression."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from hearing_ellipsoid_bench.core.types import clean_eigenvalues


@dataclass(frozen=True)
class StableWindowConfig:
    """Pre-registered admissible set and plateau-loss parameters."""

    min_window: int = 400
    start_radius: int = 25
    end_radius: int = 50
    beta: float = 1.0
    chunk_size: int = 100_000
    top_k: int = 25


@dataclass(frozen=True)
class StableWindowSearchResult:
    """Best common windows and audit tables for both supported objectives."""

    best_geometry: dict
    best_a2: dict
    top_geometry: pd.DataFrame
    top_a2: pd.DataFrame
    best_geometry_by_method: pd.DataFrame
    best_a2_by_method: pd.DataFrame
    errors_at_geometry_best: pd.DataFrame
    errors_at_a2_best: pd.DataFrame
    meta: dict


def _candidate_pairs(n: int, min_window: int) -> tuple[np.ndarray, np.ndarray]:
    starts = []
    ends = []
    for start in range(1, n - min_window + 2):
        end = np.arange(start + min_window - 1, n + 1, dtype=np.int32)
        starts.append(np.full(len(end), start, dtype=np.int32))
        ends.append(end)
    if not starts:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    return np.concatenate(starts), np.concatenate(ends)


def _prefix_normal_equations(eigs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lam = np.asarray(eigs, dtype=float)
    counts = np.arange(1, len(lam) + 1, dtype=float)
    X = np.column_stack((lam**1.5, lam, lam**0.5))
    gram_terms = X[:, :, None] * X[:, None, :]
    rhs_terms = X * counts[:, None]
    gram_prefix = np.concatenate((np.zeros((1, 3, 3)), np.cumsum(gram_terms, axis=0)))
    rhs_prefix = np.concatenate((np.zeros((1, 3)), np.cumsum(rhs_terms, axis=0)))
    return gram_prefix, rhs_prefix


def _window_relative_errors(
    eigs: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    references: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit many windows from prefix normal equations in bounded-memory chunks."""
    gram_prefix, rhs_prefix = _prefix_normal_equations(eigs)
    n_windows = len(starts)
    v_err = np.empty(n_windows, dtype=np.float64)
    s_err = np.empty(n_windows, dtype=np.float64)
    c_err = np.empty(n_windows, dtype=np.float64)
    geometry_err = np.empty(n_windows, dtype=np.float64)

    for left in range(0, n_windows, chunk_size):
        right = min(left + chunk_size, n_windows)
        s0 = starts[left:right] - 1
        e0 = ends[left:right]
        gram = gram_prefix[e0] - gram_prefix[s0]
        rhs = rhs_prefix[e0] - rhs_prefix[s0]

        scale = np.sqrt(np.diagonal(gram, axis1=1, axis2=2))
        scaled_gram = gram / scale[:, :, None] / scale[:, None, :]
        scaled_rhs = rhs / scale
        coef = np.linalg.solve(scaled_gram, scaled_rhs[..., None])[..., 0] / scale

        estimates = np.column_stack(
            (
                6.0 * np.pi**2 * coef[:, 0],
                -16.0 * np.pi * coef[:, 1],
                6.0 * np.pi**2 * coef[:, 2],
            )
        )
        rel = (estimates - references) / references
        v_err[left:right] = rel[:, 0]
        s_err[left:right] = rel[:, 1]
        c_err[left:right] = rel[:, 2]
        geometry_err[left:right] = np.sqrt(np.mean(rel**2, axis=1))

    return v_err, s_err, c_err, geometry_err


def _condition_number(eigs: np.ndarray, start: int, end: int) -> float:
    lam = eigs[start - 1 : end]
    X = np.column_stack((lam**1.5, lam, lam**0.5))
    scale = np.linalg.norm(X, axis=0)
    return float(np.linalg.cond(X / scale))


def _top_table(
    loss: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    method_loss: dict[str, np.ndarray],
    top_k: int,
    loss_name: str,
) -> pd.DataFrame:
    take = min(top_k, len(loss))
    idx = np.argpartition(loss, take - 1)[:take]
    idx = idx[np.argsort(loss[idx])]
    rows = {
        "fit_start": starts[idx],
        "fit_end": ends[idx],
        loss_name: loss[idx],
    }
    for method, values in method_loss.items():
        rows[f"{method}_loss"] = values[idx]
    return pd.DataFrame(rows)


def _near_optimal_summary(
    loss: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    relative_tolerance: float,
) -> dict:
    threshold = float(np.min(loss) * (1.0 + relative_tolerance))
    keep = loss <= threshold
    return {
        "relative_tolerance": relative_tolerance,
        "threshold": threshold,
        "window_count": int(np.count_nonzero(keep)),
        "fit_start_min": int(np.min(starts[keep])),
        "fit_start_max": int(np.max(starts[keep])),
        "fit_end_min": int(np.min(ends[keep])),
        "fit_end_max": int(np.max(ends[keep])),
    }


def _method_best_table(
    method_loss: dict[str, np.ndarray],
    starts: np.ndarray,
    ends: np.ndarray,
    loss_name: str,
) -> pd.DataFrame:
    rows = []
    for method, loss in method_loss.items():
        idx = int(np.argmin(loss))
        rows.append(
            {
                "method": method,
                "fit_start": int(starts[idx]),
                "fit_end": int(ends[idx]),
                loss_name: float(loss[idx]),
            }
        )
    return pd.DataFrame(rows)


def _errors_at_window(
    eigs_by_method: dict[str, np.ndarray],
    references: np.ndarray,
    start: int,
    end: int,
    chunk_size: int,
) -> pd.DataFrame:
    rows = []
    starts = np.array([start], dtype=np.int32)
    ends = np.array([end], dtype=np.int32)
    for method, eigs in eigs_by_method.items():
        v_err, s_err, c_err, geometry_err = _window_relative_errors(
            eigs, starts, ends, references, chunk_size=chunk_size
        )
        rows.append(
            {
                "method": method,
                "fit_start": start,
                "fit_end": end,
                "V_rel_err": float(v_err[0]),
                "S_rel_err": float(s_err[0]),
                "C_rel_err": float(c_err[0]),
                "geometry_rms_rel_err": float(geometry_err[0]),
                "scaled_design_condition": _condition_number(eigs, start, end),
            }
        )
    return pd.DataFrame(rows)


def search_stable_weyl_windows(
    eigs_by_method: dict[str, np.ndarray],
    references: tuple[float, float, float] | np.ndarray,
    *,
    config: StableWindowConfig = StableWindowConfig(),
    common_spectral_cutoff: bool = True,
) -> StableWindowSearchResult:
    """Exhaustively minimize a local plateau loss over all integer windows.

    The neighborhood is the nine-point Cartesian stencil
    ``start + {-r_s, 0, r_s}`` by ``end + {-r_e, 0, r_e}``.  For each method,
    the geometry loss is the neighborhood median RMS relative error in V, S,
    and C plus ``beta`` times the neighborhood IQR of the signed C error.  The
    common loss is the maximum method loss.  The A2-focused loss replaces the
    geometry RMS median by the median absolute C error.
    """
    if not eigs_by_method:
        raise ValueError("At least one spectrum is required.")
    if config.min_window < 3:
        raise ValueError("min_window must be at least 3.")
    if config.start_radius < 1 or config.end_radius < 1:
        raise ValueError("Neighborhood radii must be positive.")
    if config.beta < 0:
        raise ValueError("beta must be non-negative.")
    if config.chunk_size < 1 or config.top_k < 1:
        raise ValueError("chunk_size and top_k must be positive.")

    spectra = {name: clean_eigenvalues(values) for name, values in eigs_by_method.items()}
    if any(len(values) == 0 for values in spectra.values()):
        raise ValueError("Every spectrum must contain at least one positive eigenvalue.")
    references_array = np.asarray(references, dtype=float)
    if references_array.shape != (3,) or not np.all(np.isfinite(references_array)):
        raise ValueError("references must contain three finite values (V, S, C).")
    if np.any(references_array == 0):
        raise ValueError("Reference values must be nonzero for relative errors.")
    if common_spectral_cutoff:
        cutoff = min(float(values[-1]) for values in spectra.values())
        spectra = {
            name: values[values <= cutoff * (1.0 + 1e-12)]
            for name, values in spectra.items()
        }
    else:
        cutoff = None

    counts_after_cutoff = {name: int(len(values)) for name, values in spectra.items()}
    n = min(counts_after_cutoff.values())
    spectra = {name: values[:n] for name, values in spectra.items()}
    minimum_center_width = config.min_window + config.start_radius + config.end_radius
    if n < minimum_center_width + config.start_radius + config.end_radius:
        raise ValueError("Spectra are too short for the requested full neighborhood.")

    raw_starts, raw_ends = _candidate_pairs(n, config.min_window)
    center_starts, center_ends = _candidate_pairs(n, minimum_center_width)
    keep = (center_starts > config.start_radius) & (center_ends <= n - config.end_radius)
    center_starts = center_starts[keep]
    center_ends = center_ends[keep]
    common_geometry_loss = np.full(len(center_starts), -np.inf, dtype=np.float64)
    common_a2_loss = np.full(len(center_starts), -np.inf, dtype=np.float64)
    geometry_method_loss = {}
    a2_method_loss = {}
    offsets = (
        (-config.start_radius, -config.end_radius),
        (-config.start_radius, 0),
        (-config.start_radius, config.end_radius),
        (0, -config.end_radius),
        (0, 0),
        (0, config.end_radius),
        (config.start_radius, -config.end_radius),
        (config.start_radius, 0),
        (config.start_radius, config.end_radius),
    )

    for method, eigs in spectra.items():
        _, _, c_err, geometry_err = _window_relative_errors(
            eigs,
            raw_starts,
            raw_ends,
            references_array,
            chunk_size=config.chunk_size,
        )
        c_grid = np.full((n + 1, n + 1), np.nan, dtype=np.float64)
        geometry_grid = np.full((n + 1, n + 1), np.nan, dtype=np.float64)
        c_grid[raw_starts, raw_ends] = c_err
        geometry_grid[raw_starts, raw_ends] = geometry_err

        c_neighbors = np.stack(
            [c_grid[center_starts + ds, center_ends + de] for ds, de in offsets]
        )
        geometry_neighbors = np.stack(
            [geometry_grid[center_starts + ds, center_ends + de] for ds, de in offsets]
        )
        c_quartiles = np.percentile(c_neighbors, (25.0, 75.0), axis=0)
        c_iqr = c_quartiles[1] - c_quartiles[0]
        method_geometry = np.median(geometry_neighbors, axis=0) + config.beta * c_iqr
        method_a2 = np.median(np.abs(c_neighbors), axis=0) + config.beta * c_iqr
        geometry_method_loss[method] = method_geometry
        a2_method_loss[method] = method_a2
        common_geometry_loss = np.maximum(common_geometry_loss, method_geometry)
        common_a2_loss = np.maximum(common_a2_loss, method_a2)

    top_geometry = _top_table(
        common_geometry_loss,
        center_starts,
        center_ends,
        geometry_method_loss,
        config.top_k,
        "stable_geometry_loss",
    )
    top_a2 = _top_table(
        common_a2_loss,
        center_starts,
        center_ends,
        a2_method_loss,
        config.top_k,
        "stable_a2_loss",
    )
    best_geometry = top_geometry.iloc[0].to_dict()
    best_a2 = top_a2.iloc[0].to_dict()
    for best in (best_geometry, best_a2):
        best["fit_start"] = int(best["fit_start"])
        best["fit_end"] = int(best["fit_end"])
    geometry_start = int(best_geometry["fit_start"])
    geometry_end = int(best_geometry["fit_end"])
    a2_start = int(best_a2["fit_start"])
    a2_end = int(best_a2["fit_end"])

    meta = {
        "config": asdict(config),
        "methods": list(spectra),
        "common_spectral_cutoff": cutoff,
        "eigenvalue_counts_after_cutoff": counts_after_cutoff,
        "common_index_count": int(n),
        "raw_windows_evaluated_per_method": int(len(raw_starts)),
        "candidate_centers_with_full_neighborhood": int(len(center_starts)),
        "neighborhood_offsets": [list(offset) for offset in offsets],
        "geometry_near_optimal_1pct": _near_optimal_summary(
            common_geometry_loss, center_starts, center_ends, 0.01
        ),
        "a2_near_optimal_1pct": _near_optimal_summary(
            common_a2_loss, center_starts, center_ends, 0.01
        ),
    }
    return StableWindowSearchResult(
        best_geometry=best_geometry,
        best_a2=best_a2,
        top_geometry=top_geometry,
        top_a2=top_a2,
        best_geometry_by_method=_method_best_table(
            geometry_method_loss,
            center_starts,
            center_ends,
            "stable_geometry_loss",
        ),
        best_a2_by_method=_method_best_table(
            a2_method_loss,
            center_starts,
            center_ends,
            "stable_a2_loss",
        ),
        errors_at_geometry_best=_errors_at_window(
            spectra, references_array, geometry_start, geometry_end, config.chunk_size
        ),
        errors_at_a2_best=_errors_at_window(
            spectra, references_array, a2_start, a2_end, config.chunk_size
        ),
        meta=meta,
    )
