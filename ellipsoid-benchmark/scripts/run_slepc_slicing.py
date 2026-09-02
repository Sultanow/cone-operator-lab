#!/usr/bin/env python3
"""Serial SLEPc spectrum-slicing reference for an FEM ellipsoid.

The defaults reproduce the triaxial block-Hankel production operator and
derive the identical upper interval edge from the same three-term Weyl model
and ``n_target=2050``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# PETSc/SLEPc must see their command-line options before either module is used.
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

from petsc4py import PETSc
RANK = PETSc.COMM_WORLD.getRank()

def p(*a, **kw):
    """print only on rank 0"""
    if RANK == 0:
        print(*a, **kw)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run serial SLEPc spectrum slicing on an FEM ellipsoid."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.5)
    parser.add_argument("--c", type=float, default=2.3)
    parser.add_argument("--mesh-h", type=float, default=0.06)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--lambda-lo", type=float, default=0.0)
    parser.add_argument(
        "--lambda-hi",
        type=float,
        default=None,
        help="override the Weyl-derived upper edge; use 100 for a smoke test",
    )
    parser.add_argument("--n-target", type=float, default=2050.0)
    parser.add_argument("--n-tiles", type=int, default=64)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--max-it", type=int, default=100_000)
    parser.add_argument("--local-nev", type=int, default=80)
    parser.add_argument("--local-ncv", type=int, default=160)
    parser.add_argument("--partitions", type=int, default=1)
    args, _petsc_options = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    if RANK == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    volume = true_ellipsoid_volume(args.a, args.b, args.c)
    surface = ellipsoid_surface_area(args.a, args.b, args.c)
    curvature = ellipsoid_integrated_mean_curvature(args.a, args.b, args.c)
    edges, n_expected = weyl_equal_count_edges(
        args.n_tiles,
        volume,
        surface,
        curvature,
        n_target=args.n_target,
    )
    lambda_hi = float(args.lambda_hi if args.lambda_hi is not None else edges[-1])
    if args.lambda_hi is not None:
        n_expected = weyl_counting_function(lambda_hi, volume, surface, curvature)

    run_label = (
        f"a{args.a:g}_b{args.b:g}_c{args.c:g}"
        f"_P{args.order}_h{args.mesh_h:g}"
        f"_top{lambda_hi:.12g}_slepc_slicing"
    )
    p("Run label:", run_label, flush=True)
    p(f"lambda interval = [{args.lambda_lo}, {lambda_hi:.17g}]", flush=True)
    p(f"Weyl expected N = {n_expected:.6f}", flush=True)

    _, K, M, fem_meta = load_or_create_problem(
        args.data_root,
        a=args.a,
        b=args.b,
        c=args.c,
        mesh_size=args.mesh_h,
        order=args.order,
        force_remesh=False,
    )
    p(f"Free DOFs: {K.shape[0]}", flush=True)
    p(f"K nnz: {K.nnz}; M nnz: {M.nnz}", flush=True)
    p("FEM meta:", fem_meta, flush=True)

    result = solve_slepc_spectrum_slicing(
        K,
        M,
        lambda_min=args.lambda_lo,
        lambda_max=lambda_hi,
        tol=args.tol,
        max_it=args.max_it,
        local_nev=args.local_nev,
        local_ncv=args.local_ncv,
        partitions=args.partitions,
    )

    p(result, flush=True)
    if not result.success:
        raise RuntimeError(result.message)
    
    count_matches = result.meta["count_matches_inertia"]

    eig_path = args.out_dir / f"{run_label}_eigs.txt"
    if RANK == 0:
        np.savetxt(eig_path, result.eigs, fmt="%.17e")
    
        meta = {
            "algorithm": result.algorithm,
            "a": args.a,
            "b": args.b,
            "c": args.c,
            "mesh_h": args.mesh_h,
            "order": args.order,
            "free_dofs": int(K.shape[0]),
            "K_nnz": int(K.nnz),
            "M_nnz": int(M.nnz),
            "lambda_min": args.lambda_lo,
            "lambda_max": lambda_hi,
            "n_target": args.n_target,
            "n_tiles": args.n_tiles,
            "weyl_expected_count": float(n_expected),
            "n_found": result.n_found,
            "runtime_sec": float(result.runtime_sec),
            "fem_meta": {k: str(v) for k, v in (fem_meta or {}).items()},
            "solver_meta": result.meta,
        }
        meta_path = args.out_dir / f"{run_label}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    
        p("Eigenvalues:", eig_path, flush=True)
        p("Metadata:", meta_path, flush=True)
        p(
            f"SLEPc returned {result.n_found} eigenvalues; "
            f"inertia count = {result.meta['inertia_count']}; "
            f"match = {count_matches}",
            flush=True,
        )
    if count_matches is not True:
        raise RuntimeError("SLEPc eigenvalue count does not match endpoint inertia")


if __name__ == "__main__":
    main()
