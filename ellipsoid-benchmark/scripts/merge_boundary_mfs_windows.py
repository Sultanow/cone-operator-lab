#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_scale(root: Path, scale: str) -> np.ndarray:
    files = sorted((root / f"scale_{scale}").glob("window_*/*boundary_mfs_eigs.txt"))
    if not files:
        raise FileNotFoundError(f"No eigenvalue files found for scale {scale}")

    values = []
    for path in files:
        arr = np.atleast_1d(np.loadtxt(path, dtype=float))
        if arr.size:
            values.append(arr)

    if not values:
        return np.asarray([], dtype=float)

    return np.sort(np.concatenate(values))


def cluster_values(values: np.ndarray, rel_tol: float) -> np.ndarray:
    if len(values) == 0:
        return np.asarray([], dtype=float)

    values = np.sort(np.asarray(values, dtype=float))
    clusters = [[values[0]]]

    for value in values[1:]:
        center = float(np.median(clusters[-1]))
        scale = max(abs(center), abs(value), 1.0)
        if abs(value - center) <= rel_tol * scale:
            clusters[-1].append(value)
        else:
            clusters.append([value])

    return np.asarray([np.median(c) for c in clusters], dtype=float)


def stable_cross_scale(a: np.ndarray, b: np.ndarray, rel_tol: float) -> pd.DataFrame:
    rows = []
    j = 0

    for va in a:
        while j + 1 < len(b) and abs(b[j + 1] - va) < abs(b[j] - va):
            j += 1

        candidates = []
        if j < len(b):
            candidates.append(b[j])
        if j > 0:
            candidates.append(b[j - 1])
        if j + 1 < len(b):
            candidates.append(b[j + 1])

        if not candidates:
            continue

        vb = min(candidates, key=lambda x: abs(x - va))
        scale = max(abs(va), abs(vb), 1.0)
        rel_diff = abs(va - vb) / scale

        if rel_diff <= rel_tol:
            rows.append(
                {
                    "lambda_scale_1.25": va,
                    "lambda_scale_1.35": vb,
                    "lambda_stable": 0.5 * (va + vb),
                    "relative_scale_difference": rel_diff,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "lambda_scale_1.25",
                "lambda_scale_1.35",
                "lambda_stable",
                "relative_scale_difference",
            ]
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("lambda_stable").drop_duplicates(
        subset=["lambda_scale_1.35"]
    )
    return df.reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="/home/esul01/data/outputs/boundary_mfs_triaxial_windows",
    )
    p.add_argument("--within-scale-rel-tol", type=float, default=5e-5)
    p.add_argument("--cross-scale-rel-tol", type=float, default=1e-4)
    args = p.parse_args()

    root = Path(args.root)

    raw_125 = load_scale(root, "1.25")
    raw_135 = load_scale(root, "1.35")

    merged_125 = cluster_values(raw_125, args.within_scale_rel_tol)
    merged_135 = cluster_values(raw_135, args.within_scale_rel_tol)

    stable = stable_cross_scale(
        merged_125,
        merged_135,
        args.cross_scale_rel_tol,
    )

    np.savetxt(root / "mfs_scale_1.25_merged.txt", merged_125, fmt="%.16e")
    np.savetxt(root / "mfs_scale_1.35_merged.txt", merged_135, fmt="%.16e")
    np.savetxt(
        root / "mfs_cross_scale_stable.txt",
        stable["lambda_stable"].to_numpy(),
        fmt="%.16e",
    )
    stable.to_csv(root / "mfs_cross_scale_stable.csv", index=False)

    print("raw scale 1.25:", len(raw_125))
    print("raw scale 1.35:", len(raw_135))
    print("merged scale 1.25:", len(merged_125))
    print("merged scale 1.35:", len(merged_135))
    print("cross-scale stable:", len(stable))
    if len(stable):
        print("lambda range:", stable["lambda_stable"].iloc[0],
              stable["lambda_stable"].iloc[-1])
        print("median scale difference:",
              stable["relative_scale_difference"].median())
        print("max scale difference:",
              stable["relative_scale_difference"].max())


if __name__ == "__main__":
    main()
