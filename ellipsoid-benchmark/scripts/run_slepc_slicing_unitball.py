#!/usr/bin/env python3
"""Serial SLEPc spectrum-slicing reference run for the FEM unit ball."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Initialize SLEPc with the original command line before importing code that
# may import PETSc/SLEPc.  Unknown arguments below are PETSc/SLEPc options.
import slepc4py

slepc4py.init(sys.argv)

import numpy as np

from hearing_ellipsoid_bench.fem.assembly import load_or_create_problem
from hearing_ellipsoid_bench.geometry.ellipsoid import (
    ellipsoid_integrated_mean_curvature,
    ellipsoid_surface_area,
    true_ellipsoid_volume,
)
from hearing_ellipsoid_bench.solvers.fdm_block import (
    weyl_counting_function,
    weyl_equal_count_edges,
)
from hearing_ellipsoid_bench.solvers.slepc import solve_slepc_spectrum_slicing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the serial unit-ball SLEPc spectrum-slicing reference."
    )
    parser.add_argument("--data-root", type=Path, default=Path("/home/esul01/data"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--mesh-h", type=float, default=0.06)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--n-target", type=float, default=2050.0)
    parser.add_argument("--n-tiles", type=int, default=64)
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=None,
        help="override the Weyl-derived upper bound; use 100 for a smoke test",
    )
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--max-it", type=int, default=100_000)
    parser.add_argument("--local-nev", type=int, default=80)
    parser.add_argument("--local-ncv", type=int, default=160)
    args, _petsc_options = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.data_root / "outputs" / "slepc_slicing_unitball"
    out_dir.mkdir(parents=True, exist_ok=True)

    a = b = c = 1.0
    _, K, M, fem_meta = load_or_create_problem(
        args.data_root,
        a=a,
        b=b,
        c=c,
        mesh_size=args.mesh_h,
        order=args.order,
        force_remesh=False,
    )
    print("FEM meta:", fem_meta, flush=True)
    print("Free DOFs:", K.shape[0], flush=True)

    volume = true_ellipsoid_volume(a, b, c)
    surface = ellipsoid_surface_area(a, b, c)
    curvature = ellipsoid_integrated_mean_curvature(a, b, c)
    edges, n_expected = weyl_equal_count_edges(
        args.n_tiles,
        volume,
        surface,
        curvature,
        n_target=args.n_target,
    )
    lambda_min = 0.0
    lambda_max = float(args.lambda_max if args.lambda_max is not None else edges[-1])
    if args.lambda_max is not None:
        n_expected = weyl_counting_function(lambda_max, volume, surface, curvature)
    print(f"lambda interval = [{lambda_min}, {lambda_max:.12g}]", flush=True)
    print(f"Weyl expected N = {n_expected:.3f}", flush=True)

    result = solve_slepc_spectrum_slicing(
        K,
        M,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        tol=args.tol,
        max_it=args.max_it,
        local_nev=args.local_nev,
        local_ncv=args.local_ncv,
    )
    print(result, flush=True)
    if not result.success:
        raise RuntimeError(result.message)

    eig_path = out_dir / "unitball_slepc_slicing_eigs.txt"
    np.savetxt(eig_path, result.eigs, fmt="%.17e")

    meta = {
        "algorithm": result.algorithm,
        "a": a,
        "b": b,
        "c": c,
        "mesh_h": args.mesh_h,
        "order": args.order,
        "free_dofs": int(K.shape[0]),
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "weyl_expected_count": float(n_expected),
        "n_found": result.n_found,
        "runtime_sec": float(result.runtime_sec),
        "fem_meta": {k: str(v) for k, v in (fem_meta or {}).items()},
        "solver_meta": result.meta,
    }
    meta_path = out_dir / "unitball_slepc_slicing_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Eigenvalues:", eig_path, flush=True)
    print("Metadata:", meta_path, flush=True)
    print(
        f"SLEPc returned {result.n_found} eigenvalues; "
        f"inertia count = {result.meta['inertia_count']}; "
        f"match = {result.meta['count_matches_inertia']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
