from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import time
import numpy as np


@dataclass
class AlgorithmResult:
    algorithm: str
    eigs: np.ndarray
    runtime_sec: float
    success: bool = True
    message: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_found(self) -> int:
        return int(len(self.eigs))


@dataclass(frozen=True)
class GeometryConfig:
    a: float = 1.0
    b: float = 1.0
    c: float = 1.0

    @property
    def label(self) -> str:
        return f"a{self.a:g}_b{self.b:g}_c{self.c:g}"


@dataclass(frozen=True)
class FemConfig:
    mesh_size: float = 0.08
    order: int = 2
    force_remesh: bool = False


@dataclass(frozen=True)
class BenchmarkConfig:
    n_target: int
    geometry: GeometryConfig = GeometryConfig()
    fem: FemConfig = FemConfig()


def clean_eigenvalues(eigs, decimals: Optional[int] = None) -> np.ndarray:
    vals = np.asarray(eigs, dtype=float).ravel()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if decimals is not None:
        vals = np.unique(np.round(vals, decimals=decimals))
    vals.sort()
    return vals


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self.t0
