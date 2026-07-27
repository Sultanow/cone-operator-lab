#!/usr/bin/env python3
"""One (band, seed) work unit of the gap-free tile-parallel resolvent
block-Hankel scheme.

The target interval [0, lambda_top] is partitioned into disjoint
half-open Weyl-equal-count tiles, grouped into contiguous bands. Every
SLURM array task runs exactly one (band, seed) pair: one sparse LU of
(K + sigma_band M), one Chebyshev moment sequence, all tiles of the
band. Tile edges are computed HERE, deterministically, from
(a, b, c, n_target/lambda_top, n_tiles) -- the SLURM script only passes
--band-index and --seed, so runner and merge always agree on the tiling.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hearing_ellipsoid_bench.runtime.hardware import get_hardware_specs
from hearing_ellipsoid_bench.geometry.ellipsoid import (
    ellipsoid_integrated_mean_curvature,
    ellipsoid_surface_area,
    true_ellipsoid_volume,
)
from hearing_ellipsoid_bench.solvers.fdm_block import (
    band_resolvent_block_fdm,
    group_tiles_into_bands,
    weyl_equal_count_edges,
)
from hearing_ellipsoid_bench.fem.assembly import load_or_create_problem


def build_tiling(args):
    V = true_ellipsoid_volume(args.a, args.b, args.c)
    S = ellipsoid_surface_area(args.a, args.b, args.c)
    C = ellipsoid_integrated_mean_curvature(args.a, args.b, args.c)
    edges, n_total = weyl_equal_count_edges(
        args.n_tiles, V, S, C,
        lambda_top=args.lambda_top, n_target=args.n_target,
    )
    bands = group_tiles_into_bands(edges, args.n_bands)
    return V, S, C, edges, n_total, bands


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)

    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--b", type=float, default=1.5)
    p.add_argument("--c", type=float, default=2.3)
    p.add_argument("--mesh-h", type=float, default=0.06)
    p.add_argument("--order", type=int, default=2)

    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--lambda-top", type=float, default=None)
    grp.add_argument("--n-target", type=float, default=2050.0,
                     help="Expected eigenvalue count; lambda_top is "
                          "solved from the three-term Weyl law.")

    p.add_argument("--n-tiles", type=int, default=64)
    p.add_argument("--n-bands", type=int, default=6)
    p.add_argument("--band-index", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)

    p.add_argument("--n-moments", type=int, default=0,
                   help="0 = auto per band from the Weyl level spacing.")
    p.add_argument("--resolution-factor", type=float, default=0.6)
    p.add_argument("--l-max", type=int, default=34000)
    p.add_argument("--n-probes", type=int, default=13)
    p.add_argument("--hankel-blocks", type=int, default=110)
    p.add_argument("--max-rank", type=int, default=500)
    p.add_argument("--rank-tol", type=float, default=1e-10)
    p.add_argument("--transition-frac", type=float, default=0.10)
    p.add_argument("--pad-frac", type=float, default=0.75)
    p.add_argument("--sigma", type=float, default=None,
                   help="Override band shift; default sqrt(la*lb).")
    p.add_argument("--sigma-floor", type=float, default=5.0)
    p.add_argument("--circle-tol", type=float, default=0.08)
    p.add_argument("--checkpoint-every", type=int, default=2000,
                   help="Save the moment-recursion state every this many "
                        "steps; a resubmitted task resumes from it.")
    p.add_argument("--angle-cluster-frac", type=float, default=0.12,
                   help="Cluster/match tolerance as a fraction of the "
                        "local mean level spacing (in theta).")

    args = p.parse_args()
    if args.lambda_top is not None:
        args.n_target = None
    args.out_dir.mkdir(parents=True, exist_ok=True)

    V, S, C, edges, n_total, bands = build_tiling(args)
    if not (0 <= args.band_index < len(bands)):
        raise SystemExit(f"band-index must be in [0, {len(bands) - 1}]")
    tile_lo, tile_hi = bands[args.band_index]
    is_last_global = (tile_hi == args.n_tiles)

    run_label = (
        f"a{args.a:g}_b{args.b:g}_c{args.c:g}"
        f"_P{args.order}_h{args.mesh_h:g}"
        f"_top{edges[-1]:.6g}_tiles{args.n_tiles}"
        f"_band{args.band_index:02d}_seed{args.seed}"
    )
    print("Run label:", run_label, flush=True)
    print("Arguments:", vars(args), flush=True)
    print(f"Weyl geometry: V={V:.9g} S={S:.9g} C={C:.9g}", flush=True)
    print(f"lambda_top={edges[-1]:.9g}, expected N={n_total:.2f}", flush=True)
    print(f"band {args.band_index}: tiles [{tile_lo}, {tile_hi}) "
          f"lambda=[{edges[tile_lo]:.4f}, {edges[tile_hi]:.4f}]", flush=True)

    hardware = get_hardware_specs()
    t0 = time.perf_counter()

    mesh, Kmat, Mmat, fem_meta = load_or_create_problem(
        args.data_root, a=args.a, b=args.b, c=args.c,
        mesh_size=args.mesh_h, order=args.order, force_remesh=False,
    )
    print("FEM meta:", fem_meta, flush=True)
    print("Free dofs:", Kmat.shape[0], flush=True)

    ckpt_path = Path(args.out_dir) / f"{run_label}_moments_ckpt.npz"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    df, info = band_resolvent_block_fdm(
        Kmat, Mmat,
        tile_edges=edges,
        tile_lo=tile_lo,
        tile_hi=tile_hi,
        is_last_global_tile=is_last_global,
        sigma=args.sigma,
        n_moments=(args.n_moments or None),
        n_probes=args.n_probes,
        hankel_blocks=args.hankel_blocks,
        transition_frac=args.transition_frac,
        pad_frac=args.pad_frac,
        seed=args.seed,
        rank_tol=args.rank_tol,
        max_rank=args.max_rank,
        circle_tol=args.circle_tol,
        angle_cluster_frac=args.angle_cluster_frac,
        volume=V, surface=S, mean_curvature_integral=C,
        sigma_floor=args.sigma_floor,
        resolution_factor=args.resolution_factor,
        l_max=args.l_max,
        verbose=True,
        checkpoint_path=str(ckpt_path),
        checkpoint_every=args.checkpoint_every,
    )
    wall_sec = time.perf_counter() - t0

    df["band"] = args.band_index
    df["seed"] = args.seed

    cand_path = args.out_dir / f"{run_label}_candidates.csv"
    meta_path = args.out_dir / f"{run_label}_meta.json"
    df.to_csv(cand_path, index=False, float_format="%.17e")
    ckpt_path.unlink(missing_ok=True)  # moments no longer needed

    meta = {
        "run_label": run_label,
        "algorithm": "gapfree_tile_parallel_resolvent_block_hankel",
        "a": args.a, "b": args.b, "c": args.c,
        "mesh_h": args.mesh_h, "order": args.order,
        "fem_meta": {k: str(v) for k, v in fem_meta.items()},
        "free_dofs": int(Kmat.shape[0]),
        "volume": V, "surface": S, "mean_curvature_integral": C,
        "lambda_top": float(edges[-1]),
        "n_total_expected": n_total,
        "n_tiles": args.n_tiles,
        "n_bands": args.n_bands,
        "tile_edges": edges.tolist(),
        "band_index": args.band_index,
        "tile_lo": tile_lo, "tile_hi": tile_hi,
        "seed": args.seed,
        "cli_args": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
        "runtime_sec": wall_sec,
        "solver_info": info,
        "hardware": hardware,
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print("Done.", flush=True)
    print("Candidates:", cand_path, flush=True)
    print("Metadata:", meta_path, flush=True)


if __name__ == "__main__":
    main()
