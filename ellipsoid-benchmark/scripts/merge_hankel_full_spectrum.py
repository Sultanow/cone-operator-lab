#!/usr/bin/env python3
"""Merge per-(band, seed) candidate CSVs from run_hankel_band.py into a
single ordered eigenvalue spectrum WITH multiplicities, and audit its
completeness tile-by-tile against the three-term Weyl law.

Steps:
  1. Load all *_candidates.csv (columns: lambda, multiplicity, tile,
     tile_a, tile_b, band, seed).
  2. Per tile, cluster candidates across seeds by relative tolerance.
     A cluster is accepted if it is observed by >= --min-seeds seeds;
     its eigenvalue is the median lambda and its multiplicity is the
     rounded median of the per-seed multiplicity sums in the cluster.
  3. Tiles are disjoint half-open intervals, so accepted clusters
     concatenate directly -- no cross-tile deduplication is required.
  4. Audit: per tile, compare the recovered count (with multiplicity)
     to the Weyl expectation; report the largest gap-to-mean-spacing
     ratio; write tiles_to_rerun.txt for any tile with a deficit
     beyond --audit-slack.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hearing_ellipsoid_bench.geometry.ellipsoid import (
    ellipsoid_integrated_mean_curvature,
    ellipsoid_surface_area,
    true_ellipsoid_volume,
)
from hearing_ellipsoid_bench.solvers.fdm_block import (
    weyl_counting_function,
    weyl_density,
)


def load_all(root: Path, pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.rglob(pattern)):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"WARN: could not read {path}: {exc}")
            continue
        need = {"lambda", "multiplicity", "tile", "tile_a", "tile_b", "seed"}
        if not need.issubset(df.columns):
            print(f"WARN: skipping {path} (missing columns)")
            continue
        df["file"] = str(path)
        frames.append(df)
    if not frames:
        raise SystemExit(f"no candidate files below {root} matching {pattern}")
    out = pd.concat(frames, ignore_index=True)
    out = out[np.isfinite(out["lambda"])].copy()
    return out.sort_values(["tile", "lambda"]).reset_index(drop=True)


def consensus_tile(df: pd.DataFrame, spacing_frac: float, min_seeds: int,
                   density_fn=None, rel_tol_floor: float = 1e-9) -> pd.DataFrame:
    """Cluster one tile's candidates across seeds and vote.

    The clustering tolerance is SPACING-BASED: two observations belong
    to the same eigenvalue if they differ by less than
    spacing_frac * (local mean level spacing 1/rho(lambda)). Distinct
    eigenvalues are separated by about one mean spacing, so any
    spacing_frac well below 1/2 separates levels while absorbing the
    per-seed extraction scatter. density_fn(lambda) -> rho; if None, a
    fixed relative tolerance spacing_frac is used instead (legacy).
    """
    df = df.sort_values("lambda").reset_index(drop=True)
    lam = df["lambda"].to_numpy()

    def tol_at(x: float) -> float:
        if density_fn is None:
            return max(spacing_frac * max(abs(x), 1.0), rel_tol_floor)
        rho = max(float(density_fn(x)), 1e-12)
        tol = spacing_frac / rho
        # The Weyl density vanishes at the spectrum bottom, which would
        # blow the tolerance up and merge the lowest levels: cap the
        # tolerance at a small relative width.
        tol = min(tol, 2e-3 * max(abs(x), 1.0))
        return max(tol, rel_tol_floor * max(abs(x), 1.0))

    clusters, current = [], [0]

    def flush(idx):
        part = df.iloc[idx]
        values = part["lambda"].to_numpy()
        center = float(np.median(values))
        per_seed_mult = part.groupby("seed")["multiplicity"].sum()
        n_seeds = int(per_seed_mult.shape[0])
        mult = int(max(1, round(float(per_seed_mult.median()))))
        clusters.append({
            "lambda": center,
            "lambda_mean": float(np.mean(values)),
            "lambda_min": float(np.min(values)),
            "lambda_max": float(np.max(values)),
            "rel_width": float((np.max(values) - np.min(values)) / max(abs(center), 1e-30)),
            "multiplicity": mult,
            "n_seeds": n_seeds,
            "n_observations": int(len(part)),
            "tile": int(part["tile"].iloc[0]),
            "tile_a": float(part["tile_a"].iloc[0]),
            "tile_b": float(part["tile_b"].iloc[0]),
        })

    for i in range(1, len(df)):
        center = float(np.median(lam[current]))
        if abs(lam[i] - center) <= tol_at(center):
            current.append(i)
        else:
            flush(current)
            current = [i]
    flush(current)
    out = pd.DataFrame(clusters)
    out["accepted"] = out["n_seeds"] >= min_seeds
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--pattern", default="*_candidates.csv")
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--b", type=float, default=1.5)
    p.add_argument("--c", type=float, default=2.3)
    p.add_argument("--spacing-frac", type=float, default=0.15,
                   help="Cross-seed clustering tolerance as a fraction "
                        "of the local Weyl mean level spacing.")
    p.add_argument("--min-seeds", type=int, default=5)
    p.add_argument("--audit-slack", type=float, default=3.0,
                   help="Tolerated |count - Weyl| per tile before the "
                        "tile is flagged for re-running (Weyl itself "
                        "fluctuates by O(sqrt) around the smooth law).")
    p.add_argument("--gap-ratio-warn", type=float, default=4.0,
                   help="Warn when a spacing exceeds this multiple of "
                        "the local Weyl mean spacing.")
    args = p.parse_args()

    V = true_ellipsoid_volume(args.a, args.b, args.c)
    S = ellipsoid_surface_area(args.a, args.b, args.c)
    C = ellipsoid_integrated_mean_curvature(args.a, args.b, args.c)

    raw = load_all(args.root, args.pattern)
    n_seeds_avail = raw["seed"].nunique()
    print(f"raw observations: {len(raw)}  "
          f"tiles: {raw['tile'].nunique()}  seeds: {n_seeds_avail}")
    if args.min_seeds > n_seeds_avail:
        print(f"WARN: min_seeds={args.min_seeds} > available seeds "
              f"{n_seeds_avail}; lowering to {n_seeds_avail}")
        args.min_seeds = n_seeds_avail

    density = lambda lam: weyl_density(lam, V, S, C)
    per_tile = [
        consensus_tile(g, args.spacing_frac, args.min_seeds, density_fn=density)
        for _, g in raw.groupby("tile", sort=True)
    ]
    clusters = pd.concat(per_tile, ignore_index=True).sort_values("lambda").reset_index(drop=True)
    accepted = clusters[clusters["accepted"]].copy().reset_index(drop=True)

    # ---- expand by multiplicity into the final ordered spectrum ----
    spectrum = np.repeat(accepted["lambda"].to_numpy(),
                         accepted["multiplicity"].to_numpy())
    spectrum = np.sort(spectrum)

    # ---------------------- Weyl coverage audit ----------------------
    audit_rows = []
    for tile, g in accepted.groupby("tile", sort=True):
        ta, tb = float(g["tile_a"].iloc[0]), float(g["tile_b"].iloc[0])
        n_rec = int(g["multiplicity"].sum())
        n_exp = float(weyl_counting_function(tb, V, S, C)
                      - weyl_counting_function(ta, V, S, C))
        n_rej = int(clusters[(clusters["tile"] == tile) & (~clusters["accepted"])].shape[0])
        audit_rows.append({
            "tile": int(tile), "tile_a": ta, "tile_b": tb,
            "n_recovered": n_rec, "n_expected_weyl": n_exp,
            "deficit": n_exp - n_rec,
            "n_rejected_clusters": n_rej,
            "flag": abs(n_exp - n_rec) > args.audit_slack,
        })
    # tiles with zero accepted clusters would silently disappear:
    seen_tiles = set(accepted["tile"].unique())
    for tile, g in raw.groupby("tile", sort=True):
        if int(tile) not in seen_tiles:
            ta, tb = float(g["tile_a"].iloc[0]), float(g["tile_b"].iloc[0])
            n_exp = float(weyl_counting_function(tb, V, S, C)
                          - weyl_counting_function(ta, V, S, C))
            audit_rows.append({
                "tile": int(tile), "tile_a": ta, "tile_b": tb,
                "n_recovered": 0, "n_expected_weyl": n_exp,
                "deficit": n_exp, "n_rejected_clusters": 0, "flag": True,
            })
    audit = pd.DataFrame(audit_rows).sort_values("tile").reset_index(drop=True)

    # gap statistics on the final spectrum
    gap_warnings = []
    if len(spectrum) > 1:
        gaps = np.diff(spectrum)
        mids = 0.5 * (spectrum[1:] + spectrum[:-1])
        mean_sp = np.array([1.0 / max(weyl_density(m, V, S, C), 1e-12) for m in mids])
        ratio = gaps / mean_sp
        for i in np.where(ratio > args.gap_ratio_warn)[0]:
            gap_warnings.append({
                "lambda_below": float(spectrum[i]),
                "lambda_above": float(spectrum[i + 1]),
                "gap": float(gaps[i]),
                "gap_over_mean_spacing": float(ratio[i]),
            })

    # ---------------------------- output ----------------------------
    root = args.root
    clusters.to_csv(root / "full_spectrum_clusters.csv", index=False)
    accepted.to_csv(root / "full_spectrum_accepted.csv", index=False)
    audit.to_csv(root / "full_spectrum_weyl_audit.csv", index=False)
    np.savetxt(root / "full_spectrum_eigs.txt", spectrum, fmt="%.17e")

    flagged = audit[audit["flag"]]["tile"].tolist()
    (root / "tiles_to_rerun.txt").write_text(
        "\n".join(str(t) for t in flagged) + ("\n" if flagged else ""))
    (root / "full_spectrum_summary.json").write_text(json.dumps({
        "n_eigenvalues": int(len(spectrum)),
        "n_expected_weyl": float(weyl_counting_function(
            float(audit["tile_b"].max()), V, S, C)) if len(audit) else None,
        "n_accepted_clusters": int(len(accepted)),
        "n_rejected_clusters": int((~clusters["accepted"]).sum()),
        "min_seeds": args.min_seeds,
        "spacing_frac": args.spacing_frac,
        "median_cluster_rel_width": float(accepted["rel_width"].median()) if len(accepted) else None,
        "flagged_tiles": flagged,
        "gap_warnings": gap_warnings,
    }, indent=2), encoding="utf-8")

    print(f"accepted clusters: {len(accepted)}  "
          f"(rejected: {int((~clusters['accepted']).sum())})")
    print(f"final spectrum size (with multiplicity): {len(spectrum)}")
    if len(audit):
        print(f"total Weyl expectation: {audit['n_expected_weyl'].sum():.1f}  "
              f"total deficit: {audit['deficit'].sum():+.1f}")
    if flagged:
        print(f"FLAGGED tiles (|deficit| > {args.audit_slack}): {flagged}")
        print("-> re-run these tiles (see tiles_to_rerun.txt), e.g. with "
              "more moments/probes or higher max-rank.")
    else:
        print("coverage audit: no tile flagged -- spectrum is gap-free "
              "within the Weyl fluctuation tolerance.")
    if gap_warnings:
        print(f"gap warnings (> {args.gap_ratio_warn}x mean spacing): "
              f"{len(gap_warnings)} -- see full_spectrum_summary.json")


if __name__ == "__main__":
    main()
