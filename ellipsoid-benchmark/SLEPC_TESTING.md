# SLEPc handoff: smoke test and paper run

This is an overlay for the existing repository, not a replacement repository.
Copy only the included paths into the existing checkout; do not rename or
replace the checkout.  The bundle contains the serial SLEPc Krylov-Schur
spectrum-slicing reference for the FEM operators used in the paper.  Do not
start these runners with `mpiexec`: the SciPy CSR matrices are intentionally
transferred to PETSc on one MPI rank.

Example overlay installation after unpacking to a temporary directory:

```bash
rsync -av extracted/ellipsoid-benchmark/ /home/esul01/hearing-ellipsoid-bench/
```

This updates only the files contained in the overlay and leaves all Hankel,
certification, merge, notebook, and result files in the existing repository
untouched.

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

The previously certified Hankel result contains 2034 levels only up to its
largest reported eigenvalue, approximately 448.983.  SLEPc searches the full
interval through 449.1182964812976.  Therefore the SLEPc count is an
experimental result, not a prescribed acceptance target.  Both 2034 and a
larger count are scientifically meaningful.  The mandatory internal acceptance
condition is only:

```text
n_found = inertia_count
count_matches_inertia = true
```

If SLEPc reports more than 2034 levels, inspect the narrow interval
`(448.983, 449.1182964812976]` before drawing a conclusion about the top Hankel
tile.

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
reference spectrum.  The unit-ball defaults are deliberately larger than the
triaxial defaults (`local_nev=120`, `local_ncv=300`) to leave room for
multiplicities of about 55 plus neighbouring levels in one slice.

## Threading and runtime fairness

Both Slurm jobs request 16 CPUs and set `OMP_NUM_THREADS`, OpenBLAS threads and
MKL threads to 16, with OpenMP threads bound to physical cores.  Effective
MUMPS OpenMP parallelism still depends on how PETSc and MUMPS were built.  Keep
the logged environment values and verify CPU utilization during the smoke test;
otherwise a runtime comparison cannot be interpreted as a 16-core comparison.

## 5. Collect scheduler measurements

After a completed Slurm run, retain the standard-output/error logs and record
the scheduler measurements for the paper, for example:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode
```

Do not present the SLEPc result as certified unless the process exits with code
zero and the metadata contains `count_matches_inertia: true`.
