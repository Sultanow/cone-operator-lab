from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hearing_ellipsoid_bench.runtime.hardware import get_hardware_specs
from hearing_ellipsoid_bench.solvers.ssg import solve_ssg


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.5)
    parser.add_argument("--c", type=float, default=2.3)

    parser.add_argument("--l-max", type=int, default=22)
    parser.add_argument("--n-max", type=int, default=13)

    parser.add_argument("--n-radial", type=int, default=40)
    parser.add_argument("--n-theta", type=int, default=32)
    parser.add_argument("--n-phi", type=int, default=64)

    parser.add_argument("--n-eigs", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=None)

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_label = (
        f"a{args.a:g}_b{args.b:g}_c{args.c:g}"
        f"_l{args.l_max}_n{args.n_max}"
        f"_qr{args.n_radial}_qt{args.n_theta}_qp{args.n_phi}"
    )

    print("Run label:", run_label, flush=True)
    print("Arguments:", vars(args), flush=True)

    hardware = get_hardware_specs()

    result = solve_ssg(
        a=args.a,
        b=args.b,
        c=args.c,
        l_max=args.l_max,
        n_max=args.n_max,
        n_radial=args.n_radial,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
        n_eigs=args.n_eigs,
        block_size=args.block_size,
        verbose=True,
    )

    n_found = len(result.eigs)

    eig_path = args.out_dir / f"ellipsoid_{run_label}_ssg_N{n_found}_eigs.txt"
    meta_path = args.out_dir / f"ellipsoid_{run_label}_ssg_N{n_found}_meta.json"
    hardware_path = args.out_dir / f"hardware_{run_label}_ssg_N{n_found}.json"

    np.savetxt(eig_path, result.eigs, fmt="%.17e")

    meta = {
        "run_label": run_label,
        "success": bool(result.success),
        "algorithm": result.algorithm,
        "runtime_sec": float(result.runtime_sec),
        "message": getattr(result, "message", None),
        "a": args.a,
        "b": args.b,
        "c": args.c,
        "l_max": args.l_max,
        "n_max": args.n_max,
        "n_radial": args.n_radial,
        "n_theta": args.n_theta,
        "n_phi": args.n_phi,
        "n_eigs_requested": args.n_eigs,
        "n_eigs_found": n_found,
        "block_size": args.block_size,
        "solver_meta": result.meta,
        "hardware": hardware,
    }

    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    hardware_path.write_text(json.dumps(hardware, indent=2, default=str), encoding="utf-8")

    print("Done.", flush=True)
    print("Eigenvalues:", eig_path, flush=True)
    print("Metadata:", meta_path, flush=True)
    print("Hardware:", hardware_path, flush=True)

    if not result.success:
        raise RuntimeError(result.message)


if __name__ == "__main__":
    main()