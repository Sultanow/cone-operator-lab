from pathlib import Path
from hearing_ellipsoid_bench.benchmarks.reverse_geometry import run_reverse_geometry_benchmark

DATA_DIR = Path("/home/jovyan/data")
df = run_reverse_geometry_benchmark({
    "SSG": DATA_DIR / "ssg_a1_b1.5_c2.3_eigenvalues.txt",
    "Arnoldi": DATA_DIR / "ellipsoid_eigs_a1_b1.5_c2.3_arnoldi-1000.txt",
})
print(df)
