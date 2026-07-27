from pathlib import Path
from hearing_ellipsoid_bench.benchmarks.unit_ball import run_unit_ball_solver_benchmark

DATA_DIR = Path("/home/jovyan/data")
res = run_unit_ball_solver_benchmark(
    data_dir=DATA_DIR,
    truth_path=DATA_DIR / "unit_ball_N20000.txt",
    n_target=5000,
    mesh_size=0.08,
    order=2,
    solver="slepc",
)
print(res["result"])
print(res["bands"])
