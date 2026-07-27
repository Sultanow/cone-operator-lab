from __future__ import annotations

from pathlib import Path

from hearing_ellipsoid_bench.io.eigenvalues import load_eigenvalues_txt
from hearing_ellipsoid_bench.validation.weyl import reverse_geometry_table


def run_reverse_geometry_benchmark(
    eig_files: dict[str, str | Path],
    a: float = 1.0,
    b: float = 1.5,
    c: float = 2.3,
    k_values=(100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000),
):
    eigs_by_method = {
        name: load_eigenvalues_txt(path)
        for name, path in eig_files.items()
    }
    return reverse_geometry_table(eigs_by_method, a=a, b=b, c=c, k_values=k_values)
