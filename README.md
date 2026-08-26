# Hearing an Ellipsoid - reproducibility repository

This repository contains the code, spectra, meshes, notebooks, and cluster jobs for
the manuscript **Hearing an Ellipsoid**. The main benchmark compares structurally
different methods for computing large Dirichlet spectra on the triaxial ellipsoid

\[
\Omega(1,1.5,2.3)
=\left\{(x,y,z):x^2+\frac{y^2}{1.5^2}+\frac{z^2}{2.3^2}<1\right\}.
\]

The current workflow covers FEM-ARPACK, serial SLEPc spectrum slicing, Stretched
Spectral Galerkin (SSG), and the tiled resolvent block-Hankel method. Completeness of
the block-Hankel spectrum is checked independently by Sylvester inertia counts.

## Repository layout

```text
cone-operator-lab/
├── data/
│   ├── eigenvalues/                  # legacy and analytic reference spectra
│   └── ellipsoid-benchmark/
│       ├── *.msh                     # shared FEM meshes
│       ├── reference/                # exact unit-ball data
│       ├── generated/                # generated validation spectra
│       └── outputs/                  # paper spectra and result tables
├── ellipsoid-benchmark/
│   ├── src/hearing_ellipsoid_bench/  # reusable Python package
│   ├── scripts/                      # local and post-processing runners
│   ├── jobs/                         # Slurm jobs
│   ├── notebooks/                    # validation and plotting notebooks
│   ├── tests/                        # lightweight regression tests
│   └── FIT_WINDOW_OPTIMIZATION.md    # paper-ready Weyl-window methodology
├── mathematica/                      # analytic and earlier reconstruction notebooks
├── python/                           # earlier exploratory Python implementations
└── plots/                            # manuscript figures
```

The reusable implementation is in `ellipsoid-benchmark/src`. The older `python/` and
`mathematica/` directories are retained for provenance.

## Data availability

The principal triaxial spectra listed in the manuscript are committed to the repository:

| Method | Eigenvalue file | Number of values |
|---|---|---:|
| FEM-ARPACK | [`ellipsoid_a1_b1.5_c2.3_P2_h0.06_arnoldi_highacc_N2000_eigs.txt`](data/ellipsoid-benchmark/outputs/arnoldi_reference_triaxial/ellipsoid_a1_b1.5_c2.3_P2_h0.06_arnoldi_highacc_N2000_eigs.txt) | 2000 |
| SSG | [`ellipsoid_a1_b1.5_c2.3_l36_n20_qr64_qt56_qp112_ssg_N2000_eigs.txt`](data/ellipsoid-benchmark/outputs/ssg_reference_triaxial/ellipsoid_a1_b1.5_c2.3_l36_n20_qr64_qt56_qp112_ssg_N2000_eigs.txt) | 2000 |
| Certified block-Hankel | [`full_spectrum_eigs_certified.txt`](data/ellipsoid-benchmark/outputs/hankel_full_spectrum_triaxial/full_spectrum_eigs_certified.txt) | 2034 |

The FEM-ARPACK, block-Hankel, inertia, and SLEPc runs use the same quadratic FEM
pencil generated from
[`ellipsoid_a1_b1.5_c2.3_h0.06.msh`](data/ellipsoid-benchmark/ellipsoid_a1_b1.5_c2.3_h0.06.msh).
SSG is independent of the volume mesh.

Exact unit-ball reference data are available in
[`unit_ball_N20000.txt`](data/ellipsoid-benchmark/reference/unit_ball_N20000.txt).

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/Sultanow/cone-operator-lab.git
cd cone-operator-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "ellipsoid-benchmark[dev]"
```

PETSc, SLEPc, MUMPS, and their Python bindings are optional and are needed only for
the SLEPc runs. On the cluster they should be provided by the existing module or Conda
environment rather than installed through ordinary PyPI wheels.

## Quick verification

Run the tests from the benchmark directory:

```bash
cd ellipsoid-benchmark
PYTHONPATH=src python -m pytest -q
```

The tests do not require a cluster or a working SLEPc installation; the SLEPc
configuration test uses a controlled mock backend.

## Stable Weyl fit-window selection

The reverse-geometry notebook fits

\[
N(\lambda)=A_0\lambda^{3/2}+A_1\lambda+A_2\lambda^{1/2},
\qquad
V=6\pi^2A_0,\quad S=-16\pi A_1,\quad C=6\pi^2A_2.
\]

An exhaustive laptop-scale search evaluates every admissible integer fit window and a
nine-point neighborhood around it. The loss combines the neighborhood median error,
the interquartile range of the curvature error, and the worst case over FEM-ARPACK,
SSG, and block-Hankel.

```bash
cd ellipsoid-benchmark
python scripts/optimize_reverse_geometry_window.py --dry-run
```

With the committed spectra, the selected common windows are:

| Objective | `FIT_START` | `FIT_END_MAX` |
|---|---:|---:|
| Curvature-focused \(A_2/C\) validation | 31 | 524 |
| Equal-weight reconstruction of \(V,S,C\) | 32 | 720 |

The complete methodology, equations, audit values, method-specific optima, and a
paper-ready methods paragraph are in
[`ellipsoid-benchmark/FIT_WINDOW_OPTIMIZATION.md`](ellipsoid-benchmark/FIT_WINDOW_OPTIMIZATION.md).
The corresponding notebook is
[`reverse_geometry_A2_validation.ipynb`](ellipsoid-benchmark/notebooks/reverse_geometry_A2_validation.ipynb).

## SLEPc reference runs

The current SLEPc reference is intentionally **serial at the MPI level** because the
SciPy CSR matrices are transferred directly to PETSc on one rank. Do not start these
runners with `mpiexec`.

Minimal triaxial smoke test:

```bash
cd ellipsoid-benchmark
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python scripts/run_slepc_slicing.py \
  --data-root /home/esul01/data \
  --out-dir /home/esul01/data/outputs/slepc_smoke_triaxial \
  --a 1 --b 1.5 --c 2.3 --mesh-h 0.06 --order 2 \
  --lambda-hi 100 \
  -st_type sinvert \
  -st_ksp_type preonly \
  -st_pc_type cholesky \
  -st_pc_factor_mat_solver_type mumps
```

Production jobs:

```bash
sbatch jobs/run_slepc_slicing.slurm
sbatch jobs/run_slepc_slicing_unitball.slurm
```

The mandatory internal acceptance check is `count_matches_inertia = true`. The value
`2034` is not hard-coded as a required outcome: SLEPc searches through the complete
paper interval and may legitimately find an additional level above the largest
previously certified Hankel value. See
[`ellipsoid-benchmark/SLEPC_TESTING.md`](ellipsoid-benchmark/SLEPC_TESTING.md) for the
full handoff and cluster checklist.

## Block-Hankel and inertia workflow

The relevant Slurm entry points are:

```text
ellipsoid-benchmark/jobs/run_hankel_full_spectrum_array.slurm
ellipsoid-benchmark/jobs/run_certify_inertia.slurm
ellipsoid-benchmark/jobs/run_hankel_full_spectrum_array_unitball.slurm
```

The triaxial post-processing chain is implemented in:

```text
ellipsoid-benchmark/scripts/merge_hankel_full_spectrum.py
ellipsoid-benchmark/scripts/certify_inertia.py
ellipsoid-benchmark/scripts/compare_vs_reference.py
```

Do not run production SLEPc and block-Hankel jobs concurrently on the same nodes when
wall times are compared in the paper; memory-bandwidth contention would invalidate the
runtime comparison.

## Reproducibility notes

- Preserve the committed meshes and use `force_remesh=False` for cross-solver runs.
- Compare spectra at a common spectral cutoff, not merely at the same list length.
- Retain Slurm logs and `sacct` resource reports with every production result.
- Treat a fit-window optimum as conditional on its declared loss and admissible set.
- Do not describe a spectrum as certified unless the corresponding inertia checks pass.

## Additional documentation

- [Benchmark package overview](ellipsoid-benchmark/README.md)
- [Code architecture](ellipsoid-benchmark/ARCHITECTURE.md)
- [SLEPc testing and paper run](ellipsoid-benchmark/SLEPC_TESTING.md)
- [Stable fit-window optimization](ellipsoid-benchmark/FIT_WINDOW_OPTIMIZATION.md)

## Citation

If you use this repository, please cite the accompanying **Hearing an Ellipsoid**
manuscript. Formal bibliographic metadata will be added when the preprint is released.
