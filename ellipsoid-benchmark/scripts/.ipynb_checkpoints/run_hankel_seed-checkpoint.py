from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hearing_ellipsoid_bench.runtime.hardware import get_hardware_specs
from hearing_ellipsoid_bench.solvers.fdm_block import (
    windowed_resolvent_block_fdm,
)

# -------------------------------------------------------------------
# IMPORTANT:
# Reuse the same FEM assembly/load helper as in scripts/run_arnoldi_reference.py.
# If your function name differs, only adapt this import and the call below.
# -------------------------------------------------------------------
from hearing_ellipsoid_bench.fem.assembly import load_or_create_problem


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.5)
    parser.add_argument("--c", type=float, default=2.3)

    parser.add_argument("--mesh-h", type=float, default=0.06)
    parser.add_argument("--order", type=int, default=2)

    parser.add_argument("--lambda-top", type=float, default=5500.0)
    parser.add_argument("--lambda-max", type=float, default=None)

    parser.add_argument("--n-moments", type=int, default=5000)
    parser.add_argument("--n-windows", type=int, default=12)
    parser.add_argument("--n-probes", type=int, default=11)
    parser.add_argument("--hankel-blocks", type=int, default=120)
    parser.add_argument("--max-rank", type=int, default=500)

    parser.add_argument("--rank-tol", type=float, default=1e-9)
    parser.add_argument("--transition-frac", type=float, default=0.12)
    parser.add_argument("--window-overlap", type=float, default=0.20)

    parser.add_argument("--seed", type=int, required=True)
    
    parser.add_argument("--sigma", type=float, default=100.0)
    parser.add_argument("--lambda-bottom", type=float, default=0.0)

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_label = (
        f"a{args.a:g}_b{args.b:g}_c{args.c:g}"
        f"_P{args.order}_h{args.mesh_h:g}"
        f"_resolvent_sigma{args.sigma:g}"
        f"_top{args.lambda_top:g}"
        f"_mom{args.n_moments}"
        f"_win{args.n_windows}"
        f"_probe{args.n_probes}"
        f"_seed{args.seed}"
    )

    print("Run label:", run_label, flush=True)
    print("Arguments:", vars(args), flush=True)

    hardware = get_hardware_specs()

    t0 = time.perf_counter()

    mesh, Kmat, Mmat, fem_meta = load_or_create_problem(
        args.data_root,
        a=args.a,
        b=args.b,
        c=args.c,
        mesh_size=args.mesh_h,
        order=args.order,
        force_remesh=False,
    )
    
    print("FEM meta:", fem_meta, flush=True)
    print("K shape:", Kmat.shape, "nnz:", Kmat.nnz, flush=True)
    print("M shape:", Mmat.shape, "nnz:", Mmat.nnz, flush=True)
    print("Free dofs:", Kmat.shape[0], flush=True)

    eigs, info = windowed_resolvent_block_fdm(
        Kmat,
        Mmat,
        lambda_bottom=args.lambda_bottom,
        lambda_top=args.lambda_top,
        sigma=args.sigma,
        n_moments=args.n_moments,
        n_windows=args.n_windows,
        n_probes=args.n_probes,
        hankel_blocks=args.hankel_blocks,
        transition_frac=args.transition_frac,
        window_overlap=args.window_overlap,
        seed=args.seed,
        rank_tol=args.rank_tol,
        max_rank=args.max_rank,
        verbose=True,
    )

    wall_sec = time.perf_counter() - t0

    eig_path = args.out_dir / f"ellipsoid_{run_label}_hankel_candidates.txt"
    meta_path = args.out_dir / f"ellipsoid_{run_label}_hankel_meta.json"
    hardware_path = args.out_dir / f"hardware_{run_label}_hankel.json"

    np.savetxt(eig_path, np.sort(eigs), fmt="%.17e")

    meta = {
        "run_label": run_label,
        "algorithm": "multi_probe_block_hankel_fdm",
        "a": args.a,
        "b": args.b,
        "c": args.c,
        "mesh_h": args.mesh_h,
        "order": args.order,
        "fem_meta": {k: str(v) for k, v in fem_meta.items()},
        "K_shape": Kmat.shape,
        "K_nnz": int(Kmat.nnz),
        "M_shape": Mmat.shape,
        "M_nnz": int(Mmat.nnz),
        "free_dofs": int(Kmat.shape[0]),
        "lambda_top": args.lambda_top,
        "lambda_max": args.lambda_max,
        "n_moments": args.n_moments,
        "n_windows": args.n_windows,
        "n_probes": args.n_probes,
        "hankel_blocks": args.hankel_blocks,
        "max_rank": args.max_rank,
        "rank_tol": args.rank_tol,
        "transition_frac": args.transition_frac,
        "window_overlap": args.window_overlap,
        "seed": args.seed,
        "n_candidates": int(len(eigs)),
        "runtime_sec": wall_sec,
        "fdm_info": info,
        "hardware": hardware,
        "sigma": args.sigma,
        "lambda_bottom": args.lambda_bottom,
    }

    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    hardware_path.write_text(json.dumps(hardware, indent=2, default=str), encoding="utf-8")

    print("Done.", flush=True)
    print("Candidates:", eig_path, flush=True)
    print("Metadata:", meta_path, flush=True)
    print("Hardware:", hardware_path, flush=True)


if __name__ == "__main__":
    main()