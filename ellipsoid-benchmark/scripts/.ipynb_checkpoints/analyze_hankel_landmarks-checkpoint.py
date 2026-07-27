from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_values(path: Path, min_lambda: float) -> np.ndarray:
    vals = np.atleast_1d(np.loadtxt(path)).astype(float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > min_lambda]
    return np.sort(vals)


def nearest_reference(values: np.ndarray, reference: np.ndarray):
    idx = np.searchsorted(reference, values)

    idx_left = np.clip(idx - 1, 0, len(reference) - 1)
    idx_right = np.clip(idx, 0, len(reference) - 1)

    left = reference[idx_left]
    right = reference[idx_right]

    use_right = np.abs(right - values) < np.abs(left - values)

    nearest = np.where(use_right, right, left)
    nearest_idx = np.where(use_right, idx_right, idx_left)

    rel_err = (values - nearest) / nearest

    return nearest, nearest_idx, rel_err


def seed_aware_cluster(
    seed_arrays: dict[int, np.ndarray],
    rel_gap: float,
) -> pd.DataFrame:
    records = []

    for seed, vals in seed_arrays.items():
        for value in vals:
            records.append((float(value), int(seed)))

    records.sort(key=lambda x: x[0])

    if not records:
        return pd.DataFrame()

    clusters = []
    current = [records[0]]

    for value, seed in records[1:]:
        center = np.median([x[0] for x in current])

        if abs(value - center) / max(abs(center), 1e-15) <= rel_gap:
            current.append((value, seed))
        else:
            clusters.append(current)
            current = [(value, seed)]

    clusters.append(current)

    rows = []

    for cluster in clusters:
        # At most one representative value per seed.
        by_seed = {}

        provisional_center = np.median([x[0] for x in cluster])

        for value, seed in cluster:
            if seed not in by_seed:
                by_seed[seed] = value
            else:
                old = by_seed[seed]
                if abs(value - provisional_center) < abs(old - provisional_center):
                    by_seed[seed] = value

        values = np.array(list(by_seed.values()), dtype=float)
        seeds = sorted(by_seed)

        rows.append({
            "lambda_center": float(np.median(values)),
            "lambda_mean": float(np.mean(values)),
            "lambda_std": float(np.std(values)),
            "lambda_min": float(np.min(values)),
            "lambda_max": float(np.max(values)),
            "n_seeds": int(len(seeds)),
            "seeds": ",".join(map(str, seeds)),
        })

    return pd.DataFrame(rows).sort_values("lambda_center").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hankel-dir", type=Path, required=True)
    parser.add_argument("--arnoldi", type=Path, required=True)
    parser.add_argument("--ssg", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--pattern", type=str, required=True)
    parser.add_argument("--min-lambda", type=float, default=1e-8)
    parser.add_argument("--rel-gap", type=float, default=5e-4)
    parser.add_argument("--min-seeds", type=int, default=4)

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.hankel_dir.glob(args.pattern))

    seed_arrays = {}

    for path in files:
        meta_path = path.with_name(
            path.name.replace(
                "_hankel_candidates.txt",
                "_hankel_meta.json",
            )
        )

        meta = json.loads(meta_path.read_text())
        seed = int(meta["seed"])

        seed_arrays[seed] = load_values(
            path,
            min_lambda=args.min_lambda,
        )

    clusters = seed_aware_cluster(
        seed_arrays,
        rel_gap=args.rel_gap,
    )

    stable = clusters[
        clusters["n_seeds"] >= args.min_seeds
    ].copy()

    arnoldi = load_values(args.arnoldi, min_lambda=0.0)
    ssg = load_values(args.ssg, min_lambda=0.0)

    # Restrict the comparison to the common physical reference interval.
    lambda_lower = min(arnoldi[0], ssg[0])
    lambda_upper = min(arnoldi[-1], ssg[-1])
    
    stable = stable[
        (stable["lambda_center"] >= lambda_lower)
        & (stable["lambda_center"] <= lambda_upper)
    ].copy().reset_index(drop=True)

    values = stable["lambda_center"].to_numpy()

    arn_near, arn_idx, arn_err = nearest_reference(values, arnoldi)
    ssg_near, ssg_idx, ssg_err = nearest_reference(values, ssg)

    stable["arnoldi_lambda"] = arn_near
    stable["arnoldi_index"] = arn_idx + 1
    stable["arnoldi_rel_err"] = arn_err
    stable["arnoldi_abs_rel_err"] = np.abs(arn_err)

    stable["ssg_lambda"] = ssg_near
    stable["ssg_index"] = ssg_idx + 1
    stable["ssg_rel_err"] = ssg_err
    stable["ssg_abs_rel_err"] = np.abs(ssg_err)

    stable["same_reference_index"] = (
        stable["arnoldi_index"] == stable["ssg_index"]
    )

    stable.to_csv(
        args.out_dir / "hankel_seed_aware_landmarks.csv",
        index=False,
    )

    np.savetxt(
        args.out_dir / "hankel_seed_aware_landmarks.txt",
        values,
        fmt="%.17e",
    )

    summary = {
        "n_seed_files": len(files),
        "n_clusters_total": int(len(clusters)),
        "n_stable_landmarks": int(len(stable)),
        "min_seeds": args.min_seeds,
        "rel_gap": args.rel_gap,
        "lambda_min": float(values.min()) if len(values) else None,
        "lambda_max": float(values.max()) if len(values) else None,
        "median_seed_count": float(stable["n_seeds"].median())
        if len(stable) else None,
        "arnoldi_median_abs_rel_err": float(
            stable["arnoldi_abs_rel_err"].median()
        ) if len(stable) else None,
        "arnoldi_max_abs_rel_err": float(
            stable["arnoldi_abs_rel_err"].max()
        ) if len(stable) else None,
        "ssg_median_abs_rel_err": float(
            stable["ssg_abs_rel_err"].median()
        ) if len(stable) else None,
        "ssg_max_abs_rel_err": float(
            stable["ssg_abs_rel_err"].max()
        ) if len(stable) else None,
        "same_reference_index_fraction": float(
            stable["same_reference_index"].mean()
        ) if len(stable) else None,
        "arnoldi_hits_1e-3": int(
            (stable["arnoldi_abs_rel_err"] <= 1e-3).sum()
        ),
        "arnoldi_hits_1e-4": int(
            (stable["arnoldi_abs_rel_err"] <= 1e-4).sum()
        ),
        "ssg_hits_1e-3": int(
            (stable["ssg_abs_rel_err"] <= 1e-3).sum()
        ),
        "ssg_hits_1e-4": int(
            (stable["ssg_abs_rel_err"] <= 1e-4).sum()
        ),
        "lambda_reference_lower": float(lambda_lower),
        "lambda_reference_upper": float(lambda_upper),
        "stable_fraction_arnoldi_1e-3": float(
            (stable["arnoldi_abs_rel_err"] <= 1e-3).mean()
        ),
        "stable_fraction_arnoldi_1e-4": float(
            (stable["arnoldi_abs_rel_err"] <= 1e-4).mean()
        ),
        "stable_fraction_ssg_1e-3": float(
            (stable["ssg_abs_rel_err"] <= 1e-3).mean()
        ),
        "stable_fraction_ssg_1e-4": float(
            (stable["ssg_abs_rel_err"] <= 1e-4).mean()
        ),
    }

    (
        args.out_dir / "hankel_seed_aware_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))

    print("\nFirst stable landmarks:")
    print(
        stable[
            [
                "lambda_center",
                "n_seeds",
                "arnoldi_index",
                "arnoldi_abs_rel_err",
                "ssg_index",
                "ssg_abs_rel_err",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()