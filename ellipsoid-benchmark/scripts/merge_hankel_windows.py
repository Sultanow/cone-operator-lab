#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def read_candidate_file(path: Path) -> np.ndarray:
    try:
        arr = np.loadtxt(path, dtype=float)
    except Exception:
        return np.asarray([], dtype=float)

    arr = np.atleast_2d(arr)
    if arr.size == 0:
        return np.asarray([], dtype=float)

    vals = arr[:, 0].astype(float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    return vals


def parse_window_seed(path: Path) -> tuple[int | None, int | None]:
    text = str(path)
    w = re.search(r"window_(\d+)", text)
    s = re.search(r"seed[_-]?(\d+)", text)
    window = int(w.group(1)) if w else None
    seed = int(s.group(1)) if s else None
    return window, seed


def load_candidates(root: Path, pattern: str) -> pd.DataFrame:
    rows = []
    files = sorted(root.glob(pattern))

    for path in files:
        vals = read_candidate_file(path)
        window, seed = parse_window_seed(path)

        for val in vals:
            rows.append(
                {
                    "lambda": float(val),
                    "window": window,
                    "seed": seed,
                    "file": str(path),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["lambda", "window", "seed", "file"])

    return pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)


def cluster_candidates(df: pd.DataFrame, rel_tol: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("lambda").reset_index(drop=True)

    clusters = []
    current = [0]

    def flush(indices: list[int]) -> None:
        part = df.iloc[indices]
        values = part["lambda"].to_numpy()
        center = float(np.median(values))

        seeds = sorted(int(x) for x in part["seed"].dropna().unique().tolist())
        windows = sorted(int(x) for x in part["window"].dropna().unique().tolist())

        clusters.append(
            {
                "lambda_center": center,
                "lambda_mean": float(np.mean(values)),
                "lambda_min": float(np.min(values)),
                "lambda_max": float(np.max(values)),
                "rel_width": float((np.max(values) - np.min(values)) / max(abs(center), 1.0)),
                "n_observations": int(len(part)),
                "n_seeds": int(len(seeds)),
                "n_windows": int(len(windows)),
                "seeds": ",".join(map(str, seeds)),
                "windows": ",".join(map(str, windows)),
            }
        )

    for i in range(1, len(df)):
        current_center = float(np.median(df.loc[current, "lambda"]))
        value = float(df.loc[i, "lambda"])
        scale = max(abs(current_center), abs(value), 1.0)

        if abs(value - current_center) <= rel_tol * scale:
            current.append(i)
        else:
            flush(current)
            current = [i]

    flush(current)
    return pd.DataFrame(clusters).sort_values("lambda_center").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="/home/esul01/data/outputs/hankel_resolvent_triaxial_windows",
    )
    p.add_argument(
        "--pattern",
        default="window_*/seed_*/*candidate*.txt",
        help="Glob pattern below root for Hankel candidate files.",
    )
    p.add_argument("--cluster-rel-tol", type=float, default=2e-4)
    p.add_argument("--min-seeds", type=int, default=4)
    p.add_argument("--lambda-min", type=float, default=0.0)
    p.add_argument("--lambda-max", type=float, default=450.0)
    args = p.parse_args()

    root = Path(args.root)

    raw = load_candidates(root, args.pattern)
    if raw.empty:
        raise SystemExit(f"No candidates found below {root} with pattern {args.pattern}")

    raw = raw[(raw["lambda"] >= args.lambda_min) & (raw["lambda"] <= args.lambda_max)].copy()

    clusters = cluster_candidates(raw, args.cluster_rel_tol)

    stable = clusters[clusters["n_seeds"] >= args.min_seeds].copy()
    stable = stable.sort_values("lambda_center").reset_index(drop=True)

    root.mkdir(parents=True, exist_ok=True)
    raw.to_csv(root / "hankel_window_raw_candidates.csv", index=False)
    clusters.to_csv(root / "hankel_window_clusters.csv", index=False)
    stable.to_csv(root / "hankel_window_stable_clusters.csv", index=False)

    np.savetxt(
        root / "hankel_window_stable_eigs.txt",
        stable["lambda_center"].to_numpy(),
        fmt="%.16e",
    )

    print("raw observations:", len(raw))
    print("clusters:", len(clusters))
    print(f"stable clusters with >= {args.min_seeds} seeds:", len(stable))

    if len(stable):
        print("lambda range:", stable["lambda_center"].iloc[0], stable["lambda_center"].iloc[-1])
        print("median n_seeds:", stable["n_seeds"].median())
        print("min/max n_seeds:", stable["n_seeds"].min(), stable["n_seeds"].max())
        print("median rel_width:", stable["rel_width"].median())
        print("max rel_width:", stable["rel_width"].max())


if __name__ == "__main__":
    main()
