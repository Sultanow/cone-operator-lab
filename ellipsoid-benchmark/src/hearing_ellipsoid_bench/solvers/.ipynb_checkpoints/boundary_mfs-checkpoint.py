"""Boundary-only MFS / Method of Particular Solutions utilities.

This module provides an independent boundary-based eigenvalue solver for the
interior Dirichlet Helmholtz problem on ellipsoids,

    -Delta u = lambda u   in Omega(a,b,c),
             u = 0        on dOmega(a,b,c).

The implementation uses fundamental solutions with source points placed on a
scaled exterior ellipsoid. Candidate wave numbers k=sqrt(lambda) are detected
as minima of a stabilized boundary tension. The stabilization follows the
Method of Particular Solutions idea: the boundary residual is normalized by
the field sampled at interior points, avoiding meaningless minima caused only
by poorly scaled coefficient vectors.

This is intentionally named MFS rather than BEM: it is boundary-only and based
on the Helmholtz boundary representation, but it does not assemble singular
boundary-element integrals. That makes it a compact, genuinely independent
cross-check against volumetric FEM and the stretched spectral Galerkin method.

Public entry point:
    solve_mfs(...)
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from scipy.linalg import qr, svdvals
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

from hearing_ellipsoid_bench.core.types import AlgorithmResult, clean_eigenvalues


# =============================================================================
# Geometry and sampling
# =============================================================================

def fibonacci_sphere(n: int) -> np.ndarray:
    """Return approximately uniform points on the unit sphere."""
    if n < 4:
        raise ValueError("n must be at least 4")

    i = np.arange(n, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / n
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = golden_angle * i

    return np.column_stack(
        (
            radius * np.cos(phi),
            radius * np.sin(phi),
            z,
        )
    )


def ellipsoid_surface_points(
    a: float,
    b: float,
    c: float,
    n: int,
    scale: float = 1.0,
) -> np.ndarray:
    """Map Fibonacci-sphere points to a scaled ellipsoid surface."""
    if min(a, b, c) <= 0:
        raise ValueError("Ellipsoid semi-axes must be positive.")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    sphere = fibonacci_sphere(n)
    axes = np.array([a, b, c], dtype=float)
    return scale * sphere * axes[None, :]


def ellipsoid_interior_points(
    a: float,
    b: float,
    c: float,
    n: int,
    radial_min: float = 0.20,
    radial_max: float = 0.82,
) -> np.ndarray:
    """Generate deterministic interior normalization points.

    Directions come from a Fibonacci sphere; radii follow a low-discrepancy
    sequence in volume coordinates, so the points sample the ellipsoid interior
    rather than only one shell.
    """
    if n < 1:
        raise ValueError("n must be positive.")
    if not (0.0 < radial_min < radial_max < 1.0):
        raise ValueError("Require 0 < radial_min < radial_max < 1.")

    dirs = fibonacci_sphere(max(n, 4))[:n]
    j = np.arange(n, dtype=float)

    # Fractional golden-ratio sequence, transformed for approximately uniform
    # sampling in volume.
    frac = np.mod((j + 0.5) * ((np.sqrt(5.0) - 1.0) / 2.0), 1.0)
    r3_min = radial_min**3
    r3_max = radial_max**3
    radii = (r3_min + frac * (r3_max - r3_min)) ** (1.0 / 3.0)

    axes = np.array([a, b, c], dtype=float)
    return dirs * radii[:, None] * axes[None, :]


# =============================================================================
# Helmholtz fundamental-solution matrices
# =============================================================================

def pairwise_distances(targets: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Dense pairwise Euclidean distance matrix."""
    targets = np.asarray(targets, dtype=float)
    sources = np.asarray(sources, dtype=float)

    diff = targets[:, None, :] - sources[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    if np.any(dist <= 0):
        raise ValueError(
            "A target coincides with a source. Increase source_scale."
        )
    return dist


def helmholtz_green_from_distances(k: float, distances: np.ndarray) -> np.ndarray:
    """3D outgoing Helmholtz Green matrix exp(i k r)/(4 pi r)."""
    if k <= 0:
        raise ValueError("Wave number k must be positive.")

    return np.exp(1j * k * distances) / (4.0 * np.pi * distances)


@dataclass
class MFSGeometry:
    """Precomputed geometry and distance matrices for repeated k evaluations."""

    boundary_points: np.ndarray
    source_points: np.ndarray
    interior_points: np.ndarray
    boundary_source_distances: np.ndarray
    interior_source_distances: np.ndarray


def build_mfs_geometry(
    a: float,
    b: float,
    c: float,
    n_boundary: int = 420,
    n_sources: int = 300,
    n_interior: int = 180,
    source_scale: float = 1.30,
) -> MFSGeometry:
    """Build boundary, exterior-source, and interior normalization point sets."""
    if source_scale <= 1.0:
        raise ValueError(
            "source_scale must exceed 1 so all fundamental-solution sources "
            "lie outside the physical ellipsoid."
        )

    boundary = ellipsoid_surface_points(a, b, c, n_boundary, scale=1.0)
    sources = ellipsoid_surface_points(
        a, b, c, n_sources, scale=source_scale
    )
    interior = ellipsoid_interior_points(a, b, c, n_interior)

    return MFSGeometry(
        boundary_points=boundary,
        source_points=sources,
        interior_points=interior,
        boundary_source_distances=pairwise_distances(boundary, sources),
        interior_source_distances=pairwise_distances(interior, sources),
    )


# =============================================================================
# Stabilized boundary tension
# =============================================================================

def stabilized_tension(k: float, geometry: MFSGeometry) -> float:
    """Return the stabilized Dirichlet boundary tension at wave number k.

    Let A(k) sample the MFS field on the boundary and B(k) sample it at
    interior normalization points. We compute a reduced QR factorization of

        C(k) = [ A(k) ]
               [ B(k) ]

    and return sigma_min(Q_boundary), where Q_boundary is the boundary block
    of the orthonormal factor. A small value indicates a nontrivial Helmholtz
    field with small boundary trace and non-negligible interior amplitude.
    """
    A = helmholtz_green_from_distances(
        k, geometry.boundary_source_distances
    )
    B = helmholtz_green_from_distances(
        k, geometry.interior_source_distances
    )

    stacked = np.vstack((A, B))
    Q, _ = qr(stacked, mode="economic", check_finite=False)
    Q_boundary = Q[: A.shape[0], :]

    singular_values = svdvals(Q_boundary, check_finite=False)
    return float(singular_values[-1])


def scan_tension(
    geometry: MFSGeometry,
    k_min: float,
    k_max: float,
    n_scan: int,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate stabilized tension on an equidistant wave-number grid."""
    if not (0 < k_min < k_max):
        raise ValueError("Require 0 < k_min < k_max.")
    if n_scan < 5:
        raise ValueError("n_scan must be at least 5.")

    k_grid = np.linspace(k_min, k_max, n_scan)
    tensions = np.empty_like(k_grid)

    for i, k in enumerate(k_grid):
        tensions[i] = stabilized_tension(float(k), geometry)
        if verbose and (
            i == 0
            or i == n_scan - 1
            or (i + 1) % max(1, n_scan // 20) == 0
        ):
            print(
                f"  scan {i + 1:5d}/{n_scan}: "
                f"k={k:.8f}, tension={tensions[i]:.3e}",
                flush=True,
            )

    return k_grid, tensions


def refine_tension_minimum(
    geometry: MFSGeometry,
    left: float,
    right: float,
    xatol: float = 1e-10,
) -> tuple[float, float]:
    """Refine one local tension minimum by bounded scalar minimization."""
    result = minimize_scalar(
        lambda k: stabilized_tension(float(k), geometry),
        bounds=(left, right),
        method="bounded",
        options={"xatol": xatol, "maxiter": 200},
    )

    if not result.success:
        raise RuntimeError(
            f"Tension minimization failed on [{left}, {right}]: "
            f"{result.message}"
        )

    return float(result.x), float(result.fun)


def deduplicate_candidates(
    wave_numbers: np.ndarray,
    tensions: np.ndarray,
    relative_gap: float = 2e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge nearby refined candidates, retaining the lowest-tension one."""
    if len(wave_numbers) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    order = np.argsort(wave_numbers)
    ks = np.asarray(wave_numbers, dtype=float)[order]
    ts = np.asarray(tensions, dtype=float)[order]

    kept_k: list[float] = []
    kept_t: list[float] = []

    current_indices = [0]

    def flush(indices: list[int]) -> None:
        best_local = min(indices, key=lambda idx: ts[idx])
        kept_k.append(float(ks[best_local]))
        kept_t.append(float(ts[best_local]))

    for i in range(1, len(ks)):
        scale = max(abs(ks[i]), abs(ks[i - 1]), 1.0)
        if abs(ks[i] - ks[i - 1]) <= relative_gap * scale:
            current_indices.append(i)
        else:
            flush(current_indices)
            current_indices = [i]

    flush(current_indices)
    return np.asarray(kept_k), np.asarray(kept_t)


# =============================================================================
# Top-level solver
# =============================================================================

def mfs_solve(
    a: float,
    b: float,
    c: float,
    k_min: float,
    k_max: float,
    n_scan: int = 600,
    n_boundary: int = 420,
    n_sources: int = 300,
    n_interior: int = 180,
    source_scale: float = 1.30,
    tension_threshold: float = 2e-3,
    peak_prominence: float | None = None,
    relative_gap: float = 2e-4,
    refinement_xatol: float = 1e-10,
    n_eigs: int | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute distinct Dirichlet spectral levels with boundary-only MFS.

    Parameters
    ----------
    k_min, k_max:
        Search interval in wave number k=sqrt(lambda), not in lambda.
    n_scan:
        Number of coarse tension samples.
    tension_threshold:
        Retain refined local minima with tension <= this value.
    n_eigs:
        Optional maximum number of distinct levels returned after sorting.

    Returns
    -------
    eigs:
        Sorted lambda=k^2 candidates. On symmetric geometries these are
        distinct spectral levels; multiplicities are not reconstructed.
    diagnostics:
        Scan arrays, refined wave numbers, tensions, and solver metadata.
    """
    t0 = time.perf_counter()

    if verbose:
        print(f"=== Boundary MFS  a={a}, b={b}, c={c} ===")
        print(
            f"  k interval=[{k_min}, {k_max}], scan={n_scan}, "
            f"boundary={n_boundary}, sources={n_sources}, "
            f"interior={n_interior}, source_scale={source_scale}"
        )

    geometry = build_mfs_geometry(
        a=a,
        b=b,
        c=c,
        n_boundary=n_boundary,
        n_sources=n_sources,
        n_interior=n_interior,
        source_scale=source_scale,
    )

    k_grid, scan_values = scan_tension(
        geometry=geometry,
        k_min=k_min,
        k_max=k_max,
        n_scan=n_scan,
        verbose=verbose,
    )

    # Local minima of tension are peaks of -tension.
    peak_kwargs: dict[str, Any] = {}
    if peak_prominence is not None:
        peak_kwargs["prominence"] = peak_prominence

    minima, _ = find_peaks(-scan_values, **peak_kwargs)

    refined_k: list[float] = []
    refined_tension: list[float] = []

    for idx in minima:
        if idx <= 0 or idx >= len(k_grid) - 1:
            continue

        k_star, tension_star = refine_tension_minimum(
            geometry=geometry,
            left=float(k_grid[idx - 1]),
            right=float(k_grid[idx + 1]),
            xatol=refinement_xatol,
        )

        if tension_star <= tension_threshold:
            refined_k.append(k_star)
            refined_tension.append(tension_star)

    unique_k, unique_tension = deduplicate_candidates(
        np.asarray(refined_k),
        np.asarray(refined_tension),
        relative_gap=relative_gap,
    )

    eigs = unique_k**2
    order = np.argsort(eigs)
    eigs = eigs[order]
    unique_k = unique_k[order]
    unique_tension = unique_tension[order]

    if n_eigs is not None:
        eigs = eigs[:n_eigs]
        unique_k = unique_k[:n_eigs]
        unique_tension = unique_tension[:n_eigs]

    diagnostics = {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "k_min": float(k_min),
        "k_max": float(k_max),
        "n_scan": int(n_scan),
        "n_boundary": int(n_boundary),
        "n_sources": int(n_sources),
        "n_interior": int(n_interior),
        "source_scale": float(source_scale),
        "tension_threshold": float(tension_threshold),
        "peak_prominence": peak_prominence,
        "relative_gap": float(relative_gap),
        "refinement_xatol": float(refinement_xatol),
        "n_candidates": int(len(eigs)),
        "wave_numbers": unique_k,
        "tensions": unique_tension,
        "scan_k": k_grid,
        "scan_tension": scan_values,
        "runtime_sec": float(time.perf_counter() - t0),
    }

    if verbose:
        print(f"  retained candidates: {len(eigs)}")
        if len(eigs):
            print(f"  lambda range: [{eigs[0]:.12g}, {eigs[-1]:.12g}]")
            print(
                f"  tension range: "
                f"[{unique_tension.min():.3e}, {unique_tension.max():.3e}]"
            )
        print(f"  TOTAL: {diagnostics['runtime_sec']:.1f}s")

    return eigs, diagnostics


def solve_mfs(
    a: float,
    b: float,
    c: float,
    k_min: float,
    k_max: float,
    n_scan: int = 600,
    n_boundary: int = 420,
    n_sources: int = 300,
    n_interior: int = 180,
    source_scale: float = 1.30,
    tension_threshold: float = 2e-3,
    peak_prominence: float | None = None,
    relative_gap: float = 2e-4,
    refinement_xatol: float = 1e-10,
    n_eigs: int | None = None,
    verbose: bool = True,
) -> AlgorithmResult:
    """Boundary-MFS solver in the common AlgorithmResult contract."""
    name = "boundary_mfs"
    t0 = time.perf_counter()

    try:
        eigs, diagnostics = mfs_solve(
            a=a,
            b=b,
            c=c,
            k_min=k_min,
            k_max=k_max,
            n_scan=n_scan,
            n_boundary=n_boundary,
            n_sources=n_sources,
            n_interior=n_interior,
            source_scale=source_scale,
            tension_threshold=tension_threshold,
            peak_prominence=peak_prominence,
            relative_gap=relative_gap,
            refinement_xatol=refinement_xatol,
            n_eigs=n_eigs,
            verbose=verbose,
        )

        eigs = clean_eigenvalues(eigs)

        # Keep large scan arrays out of generic JSON metadata. They can be
        # exported separately by a run script if needed.
        meta = {
            key: value
            for key, value in diagnostics.items()
            if key not in {"scan_k", "scan_tension", "wave_numbers", "tensions"}
        }
        meta["wave_numbers"] = diagnostics["wave_numbers"].tolist()
        meta["tensions"] = diagnostics["tensions"].tolist()
        meta["n_eigs_requested"] = (
            None if n_eigs is None else int(n_eigs)
        )
        meta["n_eigs_found"] = int(len(eigs))

        return AlgorithmResult(
            algorithm=name,
            eigs=eigs,
            runtime_sec=time.perf_counter() - t0,
            success=True,
            meta=meta,
        )

    except Exception as exc:
        return AlgorithmResult(
            algorithm=name,
            eigs=clean_eigenvalues([]),
            runtime_sec=time.perf_counter() - t0,
            success=False,
            message=repr(exc),
            meta={
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "k_min": float(k_min),
                "k_max": float(k_max),
                "n_scan": int(n_scan),
                "n_boundary": int(n_boundary),
                "n_sources": int(n_sources),
                "n_interior": int(n_interior),
                "source_scale": float(source_scale),
                "tension_threshold": float(tension_threshold),
                "n_eigs_requested": (
                    None if n_eigs is None else int(n_eigs)
                ),
            },
        )
