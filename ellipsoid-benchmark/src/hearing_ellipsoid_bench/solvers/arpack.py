from __future__ import annotations

import time
from typing import Optional
from scipy.sparse.linalg import eigsh

from hearing_ellipsoid_bench.core.types import AlgorithmResult, clean_eigenvalues


def solve_arpack_shift_invert(K, M, n_eigs: int, sigma: float = 0.0, tol=1e-9, maxiter: Optional[int] = None) -> AlgorithmResult:
    name = "arpack_shift_invert"
    t0 = time.perf_counter()
    try:
        vals, _ = eigsh(K, k=n_eigs, M=M, sigma=sigma, which="LM", tol=tol, maxiter=maxiter)
        return AlgorithmResult(name, clean_eigenvalues(vals), time.perf_counter() - t0, meta={
            "n_eigs_requested": n_eigs,
            "sigma": sigma,
            "tol": tol,
            "n_factorizations": 1,
        })
    except Exception as e:
        return AlgorithmResult(name, clean_eigenvalues([]), time.perf_counter() - t0, False, repr(e))


def solve_arpack_windows(K, M, sigmas, chunk_size: int = 300, tol=1e-8, dedup_decimals=8) -> AlgorithmResult:
    name = "arpack_window_slicing"
    t0 = time.perf_counter()
    vals_all = []
    reports = []
    for sigma in sigmas:
        try:
            vals, _ = eigsh(K, k=chunk_size, M=M, sigma=float(sigma), which="LM", tol=tol)
            vals = clean_eigenvalues(vals)
            vals_all.extend(vals.tolist())
            reports.append({"sigma": float(sigma), "success": True, "n_found": len(vals)})
        except Exception as e:
            reports.append({"sigma": float(sigma), "success": False, "message": repr(e)})
    vals = clean_eigenvalues(vals_all, decimals=dedup_decimals)
    return AlgorithmResult(name, vals, time.perf_counter() - t0, meta={
        "chunk_size": chunk_size,
        "sigmas": list(map(float, sigmas)),
        "tol": tol,
        "window_reports": reports,
    })
