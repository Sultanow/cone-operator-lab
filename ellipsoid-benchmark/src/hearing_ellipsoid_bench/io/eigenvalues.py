from __future__ import annotations

from pathlib import Path
import numpy as np

from hearing_ellipsoid_bench.core.types import clean_eigenvalues


def load_eigenvalues_txt(path: str | Path, n: int | None = None) -> np.ndarray:
    vals = np.loadtxt(path, dtype=float)
    vals = clean_eigenvalues(vals)
    return vals[:n] if n is not None else vals


def save_eigenvalues_txt(path: str | Path, eigs, fmt: str = "%.16e") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = clean_eigenvalues(eigs)
    np.savetxt(path, vals, fmt=fmt)
    return path


def find_first_existing(paths: list[str | Path]) -> Path | None:
    for p in map(Path, paths):
        if p.exists():
            return p
    return None
