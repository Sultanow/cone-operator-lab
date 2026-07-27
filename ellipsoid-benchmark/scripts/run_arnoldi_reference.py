# scripts/run_arnoldi_reference.py

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hearing_ellipsoid_bench.fem.assembly import load_or_create_problem
from hearing_ellipsoid_bench.runtime.hardware import get_hardware_specs
from hearing_ellipsoid_bench.solvers.arnoldi_reference import (
    solve_arnoldi_reference,
    save_reference_result,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.5)
    parser.add_argument("--c", type=float, default=2.3)

    parser.add_argument("--mesh-h", type=float, default=0.06)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--n-eigs", type=int, default=3000)

    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--ncv-factor", type=float, default=2.5)
    parser.add_argument("--min-ncv", type=int, default=80)

    args = parser.parse_args()

    a, b, c = args.a, args.b, args.c
    run_label = f"a{a:g}_b{b:g}_c{c:g}_P{args.order}_h{args.mesh_h:g}"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Run label:", run_label, flush=True)
    print("Arguments:", vars(args), flush=True)

    hardware = get_hardware_specs()
    hardware_path = args.out_dir / f"hardware_{run_label}_N{args.n_eigs}.json"
    hardware_path.write_text(json.dumps(hardware, indent=2), encoding="utf-8")

    t0 = time.perf_counter()

    mesh, K, M, fem_meta = load_or_create_problem(
        args.data_root,
        a=a,
        b=b,
        c=c,
        mesh_size=args.mesh_h,
        order=args.order,
        force_remesh=False,
    )

    assembly_wall = time.perf_counter() - t0

    print("FEM meta:", fem_meta, flush=True)
    print("Assembly/load wall time [s]:", round(assembly_wall, 2), flush=True)
    print("K shape:", K.shape, "nnz:", K.nnz, flush=True)
    print("M shape:", M.shape, "nnz:", M.nnz, flush=True)
    print("Free dofs:", K.shape[0], flush=True)

    fem_info = {
        "run_label": run_label,
        "a": a,
        "b": b,
        "c": c,
        "mesh_size": args.mesh_h,
        "order": args.order,
        "target_eigs": args.n_eigs,
        "tol": args.tol,
        "maxiter": args.maxiter,
        "ncv_factor": args.ncv_factor,
        "min_ncv": args.min_ncv,
        "fem_meta": {k: str(v) for k, v in fem_meta.items()},
        "assembly_wall_sec": assembly_wall,
        "K_shape": K.shape,
        "K_nnz": int(K.nnz),
        "M_shape": M.shape,
        "M_nnz": int(M.nnz),
        "hardware": hardware,
    }

    fem_info_path = args.out_dir / f"fem_info_{run_label}_N{args.n_eigs}.json"
    fem_info_path.write_text(
        json.dumps(fem_info, indent=2, default=str),
        encoding="utf-8",
    )

    result = solve_arnoldi_reference(
        K,
        M,
        k=args.n_eigs,
        sigma=0.0,
        tol=args.tol,
        maxiter=args.maxiter,
        ncv_factor=args.ncv_factor,
        min_ncv=args.min_ncv,
        residual_batch_size=256,
    )

    eig_path, csv_path, meta_path, _ = save_reference_result(
        result,
        args.out_dir,
        run_label,
    )

    print("Done.", flush=True)
    print("Eigenvalues:", eig_path, flush=True)
    print("Residuals:", csv_path, flush=True)
    print("Metadata:", meta_path, flush=True)
    print("FEM info:", fem_info_path, flush=True)
    print("Hardware:", hardware_path, flush=True)


if __name__ == "__main__":
    main()