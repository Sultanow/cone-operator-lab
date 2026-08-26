#!/usr/bin/env python3
"""Reproduce the stable FIT_START/FIT_END_MAX search used in the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hearing_ellipsoid_bench.geometry.ellipsoid import (
    ellipsoid_integrated_mean_curvature,
    ellipsoid_surface_area,
    true_ellipsoid_volume,
)
from hearing_ellipsoid_bench.validation.window_search import (
    StableWindowConfig,
    search_stable_weyl_windows,
)


FILENAMES = {
    "arnoldi": (
        "arnoldi_reference_triaxial/"
        "ellipsoid_a1_b1.5_c2.3_P2_h0.06_arnoldi_highacc_N2000_eigs.txt"
    ),
    "ssg": (
        "ssg_reference_triaxial/"
        "ellipsoid_a1_b1.5_c2.3_l36_n20_qr64_qt56_qp112_ssg_N2000_eigs.txt"
    ),
    "hankel": "hankel_full_spectrum_triaxial/full_spectrum_eigs_certified.txt",
}


def find_data_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [
        root / "data" / "ellipsoid-benchmark" for root in (cwd, *cwd.parents)
    ]
    candidates.append(Path.home() / "data")
    for candidate in candidates:
        if all((candidate / "outputs" / filename).is_file() for filename in FILENAMES.values()):
            return candidate
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not locate all three spectra. Searched:\n{searched}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-window", type=int, default=400)
    parser.add_argument("--start-radius", type=int, default=25)
    parser.add_argument("--end-radius", type=int, default=50)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true", help="compute but do not write files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = (args.data_root or find_data_root()).resolve()
    input_paths = {
        method: data_root / "outputs" / filename for method, filename in FILENAMES.items()
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing spectra:\n  - " + "\n  - ".join(missing))

    eigs_by_method = {method: np.loadtxt(path) for method, path in input_paths.items()}
    a, b, c = 1.0, 1.5, 2.3
    references = (
        true_ellipsoid_volume(a, b, c),
        ellipsoid_surface_area(a, b, c),
        ellipsoid_integrated_mean_curvature(a, b, c, n_theta=240, n_phi=480),
    )
    config = StableWindowConfig(
        min_window=args.min_window,
        start_radius=args.start_radius,
        end_radius=args.end_radius,
        beta=args.beta,
        top_k=args.top_k,
    )
    result = search_stable_weyl_windows(eigs_by_method, references, config=config)

    summary = {
        "geometry": {"a": a, "b": b, "c": c},
        "references": {"V": references[0], "S": references[1], "C": references[2]},
        "input_files": {method: str(path) for method, path in input_paths.items()},
        "best_geometry": result.best_geometry,
        "best_a2": result.best_a2,
        "meta": result.meta,
    }
    print(json.dumps(summary, indent=2))
    print("\nErrors at geometry optimum:")
    print(result.errors_at_geometry_best.to_string(index=False))
    print("\nErrors at A2 optimum:")
    print(result.errors_at_a2_best.to_string(index=False))
    print("\nMethod-specific stable geometry optima:")
    print(result.best_geometry_by_method.to_string(index=False))
    print("\nMethod-specific stable A2 optima:")
    print(result.best_a2_by_method.to_string(index=False))

    if args.dry_run:
        return

    out_dir = (args.out_dir or data_root / "outputs" / "reverse_geometry_a2").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fit_window_optimization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    result.top_geometry.to_csv(out_dir / "fit_window_top_geometry.csv", index=False)
    result.top_a2.to_csv(out_dir / "fit_window_top_a2.csv", index=False)
    result.best_geometry_by_method.to_csv(
        out_dir / "fit_window_best_geometry_by_method.csv", index=False
    )
    result.best_a2_by_method.to_csv(
        out_dir / "fit_window_best_a2_by_method.csv", index=False
    )
    result.errors_at_geometry_best.to_csv(
        out_dir / "fit_window_errors_geometry_best.csv", index=False
    )
    result.errors_at_a2_best.to_csv(out_dir / "fit_window_errors_a2_best.csv", index=False)
    print(f"\nWrote audit files to {out_dir}")


if __name__ == "__main__":
    main()
