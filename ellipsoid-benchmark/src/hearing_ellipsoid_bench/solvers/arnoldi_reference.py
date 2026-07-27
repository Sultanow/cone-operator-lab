# src/hearing_ellipsoid_bench/solvers/arnoldi_reference.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh

from hearing_ellipsoid_bench.core.types import clean_eigenvalues


@dataclass
class ArnoldiReferenceResult:
    k_requested: int
    eigvals: np.ndarray
    residual_abs: np.ndarray
    residual_rel: np.ndarray
    wall_sec: float
    meta: dict


def choose_ncv(n_dofs: int, k: int, factor: float = 2.5, min_ncv: int = 80) -> int:
    """Choose ARPACK ncv safely. ARPACK requires k < ncv <= n."""
    ncv = int(max(min_ncv, np.ceil(factor * k)))
    ncv = min(ncv, max(k + 2, n_dofs - 1))
    return ncv


def compute_generalized_residuals(K, M, vals, vecs, batch_size: int = 256):
    """Compute ||K v - lambda M v|| and divide by |lambda|."""
    vals = np.asarray(vals, dtype=float)
    n = len(vals)

    abs_res = np.empty(n, dtype=float)
    rel_res = np.empty(n, dtype=float)

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)

        V = vecs[:, start:stop]
        KV = K @ V
        MV = M @ V
        R = KV - MV * vals[start:stop][None, :]

        abs_chunk = np.linalg.norm(R, axis=0)
        abs_res[start:stop] = abs_chunk
        rel_res[start:stop] = abs_chunk / np.maximum(np.abs(vals[start:stop]), 1e-300)

    return abs_res, rel_res


def solve_arnoldi_reference(
    K,
    M,
    k: int,
    sigma: float = 0.0,
    tol: float = 1e-10,
    maxiter=None,
    ncv_factor: float = 2.5,
    min_ncv: int = 80,
    residual_batch_size: int = 256,
) -> ArnoldiReferenceResult:
    n_dofs = K.shape[0]

    if k >= n_dofs - 1:
        raise ValueError(f"k={k} is too large for n_dofs={n_dofs}. Need k < n_dofs - 1.")

    ncv = choose_ncv(n_dofs, k, factor=ncv_factor, min_ncv=min_ncv)

    print(
        f"Running ARPACK shift-invert: k={k}, sigma={sigma}, "
        f"tol={tol}, ncv={ncv}, n_dofs={n_dofs}",
        flush=True,
    )

    t0 = time.perf_counter()

    vals, vecs = eigsh(
        K,
        k=k,
        M=M,
        sigma=sigma,
        which="LM",
        tol=tol,
        maxiter=maxiter,
        ncv=ncv,
        return_eigenvectors=True,
    )

    wall = time.perf_counter() - t0

    order = np.argsort(vals)
    vals = np.asarray(vals[order], dtype=float)
    vecs = vecs[:, order]

    print(f"Computing residuals for {len(vals)} Ritz pairs ...", flush=True)
    abs_res, rel_res = compute_generalized_residuals(
        K,
        M,
        vals,
        vecs,
        batch_size=residual_batch_size,
    )

    meta = {
        "solver": "scipy.sparse.linalg.eigsh",
        "mode": "shift-invert",
        "sigma": sigma,
        "which": "LM",
        "tol": tol,
        "maxiter": maxiter,
        "ncv_factor": float(ncv_factor),
        "min_ncv": int(min_ncv),
        "ncv": int(ncv),
        "n_dofs": int(n_dofs),
        "wall_sec": float(wall),
        "lambda_min": float(vals[0]),
        "lambda_max": float(vals[-1]),
        "residual_abs_max": float(np.max(abs_res)),
        "residual_abs_median": float(np.median(abs_res)),
        "residual_rel_max": float(np.max(rel_res)),
        "residual_rel_median": float(np.median(rel_res)),
    }

    del vecs

    return ArnoldiReferenceResult(
        k_requested=k,
        eigvals=clean_eigenvalues(vals),
        residual_abs=abs_res,
        residual_rel=rel_res,
        wall_sec=wall,
        meta=meta,
    )


def save_reference_result(
    result: ArnoldiReferenceResult,
    out_dir: Path,
    run_label: str,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k = result.k_requested
    stem = f"ellipsoid_{run_label}_arnoldi_highacc_N{k}"

    eig_path = out_dir / f"{stem}_eigs.txt"
    csv_path = out_dir / f"{stem}_residuals.csv"
    meta_path = out_dir / f"{stem}_meta.json"

    np.savetxt(eig_path, result.eigvals, fmt="%.17e")

    df = pd.DataFrame(
        {
            "k": np.arange(1, len(result.eigvals) + 1),
            "lambda": result.eigvals,
            "residual_abs": result.residual_abs,
            "residual_rel": result.residual_rel,
        }
    )
    df.to_csv(csv_path, index=False)

    meta_path.write_text(
        json.dumps(result.meta, indent=2, default=str),
        encoding="utf-8",
    )

    print("saved eigenvalues:", eig_path, flush=True)
    print("saved residuals:", csv_path, flush=True)
    print("saved metadata:", meta_path, flush=True)

    return eig_path, csv_path, meta_path, df