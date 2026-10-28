# SLEPc spectrum slicing: smoke test and paper run

This document describes the SLEPc Krylov-Schur spectrum-slicing reference for the
FEM operators used in the paper (Sections 3.4 and 5.3). The runs at $h=0.06$ are
serial at the MPI level: the SciPy CSR matrices are intentionally transferred to PETSc
on one MPI rank, so do not start them with `mpiexec`. The refined-mesh runs at
$h=0.03$ (Section 4 below) are the exception and run under MPI.

All commands below are run from the `ellipsoid-benchmark` directory of a clone of
this repository. The meshes and outputs live in `../data/ellipsoid-benchmark`:

```bash
cd cone-operator-lab/ellipsoid-benchmark
export DATA_ROOT="$(realpath ../data/ellipsoid-benchmark)"
```

## 1. Environment check

Activate the cluster environment that provides PETSc, SLEPc and MUMPS, then
install the package in editable mode:

```bash
conda activate slepc-env
python -m pip install -e .
python -c "import numpy, scipy, meshio, skfem, petsc4py, slepc4py; print('imports OK')"
```

The FEM meshes must be available in `$DATA_ROOT`. In particular, the paper run uses
`a=1`, `b=1.5`, `c=2.3`, `h=0.06`, and quadratic elements.

The Slurm jobs in `jobs/` contain the paths of the cluster on which the paper runs
were executed. Before submitting, adapt the log directory (`#SBATCH --output`,
`#SBATCH --error`), the Conda environment, the repository path and `--data-root`.

## 2. Minimal triaxial smoke test

Run a small interval before requesting the full allocation:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python scripts/run_slepc_slicing.py \
  --data-root "$DATA_ROOT" \
  --out-dir "$DATA_ROOT/outputs/slepc_smoke_triaxial" \
  --a 1 --b 1.5 --c 2.3 --mesh-h 0.06 --order 2 \
  --lambda-hi 100 \
  -st_type sinvert \
  -st_ksp_type preonly \
  -st_pc_type cholesky \
  -st_pc_factor_mat_solver_type mumps \
  -st_mat_mumps_icntl_13 1
```

The final line must report `match = True`. Use a separate output directory for
the smoke test so it cannot overwrite production output.

## 3. Triaxial paper run

```bash
sbatch jobs/run_slepc_slicing.slurm
```

The runner derives the interval endpoint from the same three-term Weyl model used by
the block-Hankel run:

```text
lambda_top = 449.1182964812976
Weyl expected count = 2050
```

The certified block-Hankel result contains 2034 levels up to its largest reported
eigenvalue, approximately 448.983, whereas SLEPc searches the full interval through
449.1182964812976. The SLEPc count is therefore a result, not a prescribed acceptance
target. The mandatory internal acceptance condition is only:

```text
n_found = inertia_count
count_matches_inertia = true
```

The JSON metadata also records the SLEPc convergence reason, maximum and median
relative residual, all slicing shifts and inertias, matrix dimensions and
nonzero counts, and the exact FEM mesh path.

**Result in the paper:** 2034 eigenvalues in 18:52 min on 16 cores (4.9 core-hours),
maximum relative residual $7.2\times10^{-13}$, count confirmed by inertia
([`outputs/slepc_slicing_triaxial`](../data/ellipsoid-benchmark/outputs/slepc_slicing_triaxial)).

## 4. Refined mesh ($h=0.03$)

```bash
sbatch jobs/run_slepc_slicing_h003_i64.slurm
```

At $h=0.03$ the sparse factors hold about $6.3\times10^{9}$ entries, beyond the
$2^{31}$ limit of a PETSc build with 32-bit indices. Such a build does not report an
error; it ends in a segmentation fault after about ninety minutes. PETSc and SLEPc
must therefore be built with 64-bit indices (paper, Listing 2), and the job sets
`PETSC_DIR` and `SLEPC_DIR` to that build.

Unlike the $h=0.06$ runs, this job starts the runner with `mpirun` on 32 ranks split
into two spectrum partitions (`--partitions 2`). Memory grows with the number of
partitions, since each holds its own factorization: two partitions fit into a 700 GB
node, eight do not.

**Result in the paper:** 2050 eigenvalues, the Weyl prediction to the digit, in
5:41 hours on 64 cores, confirmed by inertia
([`outputs/slepc_slicing_triaxial_h003_i64`](../data/ellipsoid-benchmark/outputs/slepc_slicing_triaxial_h003_i64)).

## 5. Unit-ball validation

First use `--lambda-max 100` with `scripts/run_slepc_slicing_unitball.py`, or
submit the complete runs with:

```bash
sbatch jobs/run_slepc_slicing_unitball.slurm        # h = 0.06
sbatch jobs/run_slepc_slicing_unitball_h003.slurm   # h = 0.03, 64-bit PETSc
```

`run_slepc_slicing_unitball.py` defaults to a cluster-specific `--data-root`; pass
`--data-root "$DATA_ROOT"` when running it by hand.

Again, the mandatory internal check is `count_matches_inertia = true`. The resulting
eigenvalues can then be compared with the analytical spherical-Bessel reference
spectrum using `scripts/validate_unitball_multiplicities.py`. The unit-ball defaults
are deliberately larger than the triaxial defaults (`local_nev=120`, `local_ncv=300`)
to leave room for multiplicities of about 55 plus neighbouring levels in one slice.

## Threading and runtime fairness

The 16-core Slurm jobs set `OMP_NUM_THREADS`, OpenBLAS threads and MKL threads to 16,
with OpenMP threads bound to physical cores. Effective MUMPS OpenMP parallelism still
depends on how PETSc and MUMPS were built. Keep the logged environment values and
verify CPU utilization during the smoke test; otherwise a runtime comparison cannot
be interpreted as a 16-core comparison.

## 6. Collect scheduler measurements

After a completed Slurm run, retain the standard-output/error logs and record
the scheduler measurements, for example:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode
```

Do not present the SLEPc result as certified unless the process exits with code
zero and the metadata contains `count_matches_inertia: true`.
