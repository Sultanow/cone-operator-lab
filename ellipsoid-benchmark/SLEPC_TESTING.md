# SLEPc handoff: smoke test and paper run

This bundle contains the serial SLEPc Krylov-Schur spectrum-slicing reference
for the FEM operators used in the paper.  Do not start these runners with
`mpiexec`: the SciPy CSR matrices are intentionally transferred to PETSc on one
MPI rank.

## 1. Environment check

Activate the cluster environment that provides PETSc, SLEPc and MUMPS, then
install the package in editable mode:

```bash
conda activate slepc-env
cd /home/esul01/hearing-ellipsoid-bench
python -m pip install -e .
python -c "import numpy, scipy, meshio, skfem, petsc4py, slepc4py; print('imports OK')"
```

The existing FEM meshes must be available below `/home/esul01/data`.  In
particular, the paper run uses `a=1`, `b=1.5`, `c=2.3`, `h=0.06`, and quadratic
elements.  If the environment name or repository path differs, update the two
files in `jobs/` before submission.

## 2. Minimal triaxial smoke test

Run a small interval before requesting the full allocation:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python scripts/run_slepc_slicing.py \
  --data-root /home/esul01/data \
  --out-dir /home/esul01/data/outputs/slepc_smoke_triaxial \
  --a 1 --b 1.5 --c 2.3 --mesh-h 0.06 --order 2 \
  --lambda-hi 100 \
  -st_type sinvert \
  -st_ksp_type preonly \
  -st_pc_type cholesky \
  -st_pc_factor_mat_solver_type mumps \
  -st_mat_mumps_icntl_13 1
```

The final line must report `match = True`.  Use a separate output directory for
the smoke test so it cannot overwrite production output.

## 3. Triaxial paper run

```bash
sbatch jobs/run_slepc_slicing.slurm
```

The runner derives the paper interval endpoint from the same three-term Weyl
model used by the block-Hankel run:

```text
lambda_top = 449.1182964812976
Weyl expected count = 2050
```

The paper-level acceptance target is:

```text
n_found = 2034
inertia_count = 2034
count_matches_inertia = true
```

The JSON metadata also records the SLEPc convergence reason, maximum and median
relative residual, all slicing shifts and inertias, matrix dimensions and
nonzero counts, and the exact FEM mesh path.

## 4. Unit-ball validation

First use `--lambda-max 100` with `scripts/run_slepc_slicing_unitball.py`, or
submit the complete run with:

```bash
sbatch jobs/run_slepc_slicing_unitball.slurm
```

Again, the mandatory internal check is `count_matches_inertia = true`.  The
resulting eigenvalues can then be compared with the analytical spherical-Bessel
reference spectrum.

## 5. Collect scheduler measurements

After a completed Slurm run, retain the standard-output/error logs and record
the scheduler measurements for the paper, for example:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode
```

Do not present the SLEPc result as certified unless the process exits with code
zero and the metadata contains `count_matches_inertia: true`.
