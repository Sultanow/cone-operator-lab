from __future__ import annotations

from pathlib import Path
import pandas as pd

from hearing_ellipsoid_bench.io.eigenvalues import load_eigenvalues_txt, save_eigenvalues_txt
from hearing_ellipsoid_bench.fem.assembly import load_or_create_problem
from hearing_ellipsoid_bench.solvers.slepc import solve_slepc_krylov_schur
from hearing_ellipsoid_bench.solvers.arpack import solve_arpack_shift_invert
from hearing_ellipsoid_bench.validation.sphere import compare_to_truth, error_bands


def run_unit_ball_solver_benchmark(
    data_dir: str | Path,
    truth_path: str | Path,
    n_target: int = 5000,
    mesh_size: float = 0.08,
    order: int = 2,
    solver: str = "slepc",
    force_remesh: bool = False,
):
    """Run a clean unit-ball benchmark against known sphere eigenvalues."""
    data_dir = Path(data_dir)
    truth = load_eigenvalues_txt(truth_path, n=n_target)

    _, K, M, fem_meta = load_or_create_problem(
        data_dir,
        a=1.0,
        b=1.0,
        c=1.0,
        mesh_size=mesh_size,
        order=order,
        force_remesh=force_remesh,
    )

    if solver == "slepc":
        result = solve_slepc_krylov_schur(K, M, n_eigs=n_target)
    elif solver == "arpack":
        result = solve_arpack_shift_invert(K, M, n_eigs=n_target)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    df = compare_to_truth(result.eigs, truth, n=n_target, label=result.algorithm)
    bands = error_bands(df)

    out_dir = data_dir / "benchmark_outputs_clean"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_eigenvalues_txt(out_dir / f"unit_ball_P{order}_h{mesh_size:g}_{result.algorithm}_eigs.txt", result.eigs)
    df.to_csv(out_dir / f"unit_ball_P{order}_h{mesh_size:g}_{result.algorithm}_compare.csv", index=False)
    bands.to_csv(out_dir / f"unit_ball_P{order}_h{mesh_size:g}_{result.algorithm}_bands.csv", index=False)

    return {"result": result, "compare": df, "bands": bands, "fem_meta": fem_meta}
