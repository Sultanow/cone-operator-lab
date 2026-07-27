#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hearing_ellipsoid_bench.solvers.boundary_mfs import solve_mfs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--a", type=float, required=True)
    p.add_argument("--b", type=float, required=True)
    p.add_argument("--c", type=float, required=True)
    p.add_argument("--k-min", type=float, required=True)
    p.add_argument("--k-max", type=float, required=True)
    p.add_argument("--n-scan", type=int, default=500)
    p.add_argument("--n-boundary", type=int, default=420)
    p.add_argument("--n-sources", type=int, default=300)
    p.add_argument("--n-interior", type=int, default=180)
    p.add_argument("--source-scale", type=float, default=1.30)
    p.add_argument("--tension-threshold", type=float, default=2e-3)
    p.add_argument("--relative-gap", type=float, default=2e-4)
    p.add_argument("--refinement-xatol", type=float, default=1e-10)
    p.add_argument("--n-eigs", type=int, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = solve_mfs(
        a=args.a,
        b=args.b,
        c=args.c,
        k_min=args.k_min,
        k_max=args.k_max,
        n_scan=args.n_scan,
        n_boundary=args.n_boundary,
        n_sources=args.n_sources,
        n_interior=args.n_interior,
        source_scale=args.source_scale,
        tension_threshold=args.tension_threshold,
        relative_gap=args.relative_gap,
        refinement_xatol=args.refinement_xatol,
        n_eigs=args.n_eigs,
        verbose=True,
    )

    tag = (
        f"a{args.a:g}_b{args.b:g}_c{args.c:g}"
        f"_k{args.k_min:g}-{args.k_max:g}"
        f"_scan{args.n_scan}"
        f"_bdry{args.n_boundary}"
        f"_src{args.n_sources}"
        f"_int{args.n_interior}"
        f"_scale{args.source_scale:g}"
    )

    eig_path = out_dir / f"ellipsoid_{tag}_boundary_mfs_eigs.txt"
    meta_path = out_dir / f"ellipsoid_{tag}_boundary_mfs_meta.json"

    np.savetxt(eig_path, result.eigs, fmt="%.16e")

    meta = dict(result.meta or {})
    meta.update(
        {
            "algorithm": result.algorithm,
            "runtime_sec": float(result.runtime_sec),
            "success": bool(result.success),
            "message": result.message,
            "eigenvalue_file": str(eig_path),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("success:", result.success)
    print("n_eigs:", len(result.eigs))
    if len(result.eigs):
        print("lambda_1:", result.eigs[0])
        print("lambda_last:", result.eigs[-1])
    print("saved:", eig_path)
    print("saved:", meta_path)

    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
