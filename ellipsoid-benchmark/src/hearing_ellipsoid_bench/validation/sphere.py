from __future__ import annotations

import numpy as np
import pandas as pd

from hearing_ellipsoid_bench.core.types import clean_eigenvalues


def compare_to_truth(eigs_num, eigs_true, n: int | None = None, label: str = "num") -> pd.DataFrame:
    num = clean_eigenvalues(eigs_num)
    true = clean_eigenvalues(eigs_true)
    if n is None:
        n = min(len(num), len(true))
    n = min(n, len(num), len(true))
    df = pd.DataFrame({
        "k": np.arange(1, n + 1),
        "lambda_true": true[:n],
        "lambda_num": num[:n],
        "method": label,
    })
    df["delta"] = df["lambda_num"] - df["lambda_true"]
    df["abs_delta"] = np.abs(df["delta"])
    df["rel_err"] = df["delta"] / df["lambda_true"]
    df["abs_rel_err"] = np.abs(df["rel_err"])
    return df


def error_bands(
    df: pd.DataFrame,
    bands=((1, 50), (51, 100), (101, 200), (201, 500), (501, 1000),
           (1001, 2000), (2001, 3000), (3001, 4000), (4001, 5000)),
) -> pd.DataFrame:
    rows = []
    for lo, hi in bands:
        part = df[(df["k"] >= lo) & (df["k"] <= hi)]
        if part.empty:
            continue
        rows.append({
            "k_range": f"{lo}-{hi}",
            "count": len(part),
            "median_abs_rel_err": part["abs_rel_err"].median(),
            "mean_abs_rel_err": part["abs_rel_err"].mean(),
            "max_abs_rel_err": part["abs_rel_err"].max(),
        })
    return pd.DataFrame(rows)


def cluster_true_eigenvalues(eigs_true, n_target: int, rel_tol=1e-10, abs_tol=1e-10) -> pd.DataFrame:
    vals = clean_eigenvalues(eigs_true)[:n_target]
    clusters = []
    start = 0
    for i in range(1, len(vals)):
        tol = max(abs_tol, rel_tol * max(abs(vals[i - 1]), abs(vals[i])))
        if abs(vals[i] - vals[i - 1]) > tol:
            part = vals[start:i]
            clusters.append({
                "cluster_id": len(clusters) + 1,
                "k_start": start + 1,
                "k_end": i,
                "multiplicity": i - start,
                "lambda_min": float(part.min()),
                "lambda_max": float(part.max()),
                "lambda_center": float(part.mean()),
            })
            start = i
    part = vals[start:]
    clusters.append({
        "cluster_id": len(clusters) + 1,
        "k_start": start + 1,
        "k_end": len(vals),
        "multiplicity": len(vals) - start,
        "lambda_min": float(part.min()),
        "lambda_max": float(part.max()),
        "lambda_center": float(part.mean()),
    })
    return pd.DataFrame(clusters)


def make_cluster_windows(eigs_true, n_target: int, clusters_per_window=4, pad_rel=1e-6) -> pd.DataFrame:
    dfc = cluster_true_eigenvalues(eigs_true, n_target)
    rows = []
    for start in range(0, len(dfc), clusters_per_window):
        part = dfc.iloc[start:start + clusters_per_window]
        left = float(part["lambda_min"].min())
        right = float(part["lambda_max"].max())
        pad = pad_rel * max(1.0, abs(left), abs(right))
        rows.append({
            "window_id": len(rows) + 1,
            "lambda_left": left - pad,
            "lambda_right": right + pad,
            "k_start": int(part["k_start"].min()),
            "k_end": int(part["k_end"].max()),
            "expected_count": int(part["multiplicity"].sum()),
        })
    return pd.DataFrame(rows)


def inflate_windows(df_windows: pd.DataFrame, rel_pad=0.10, abs_pad=1e-8) -> pd.DataFrame:
    out = df_windows.copy()
    centers = 0.5 * (out["lambda_left"] + out["lambda_right"])
    half = 0.5 * (out["lambda_right"] - out["lambda_left"])
    extra = np.maximum(abs_pad, rel_pad * centers.abs())
    out["lambda_left_original"] = out["lambda_left"]
    out["lambda_right_original"] = out["lambda_right"]
    out["lambda_left"] = centers - half - extra
    out["lambda_right"] = centers + half + extra
    out["window_inflation_rel"] = rel_pad
    return out
