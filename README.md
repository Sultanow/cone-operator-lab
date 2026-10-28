# Hearing Ellipsoids – reproducibility repository

[![AppliedMath](https://img.shields.io/badge/AppliedMath-6%2811%29%2C%20179%20%282026%29-orange)](https://www.mdpi.com/2673-9909/6/11/179) [![DOI](https://img.shields.io/badge/DOI-10.3390%2Fappliedmath6110179-blue)](https://doi.org/10.3390/appliedmath6110179) [![Open Access](https://img.shields.io/badge/Paper-open%20access-lightgrey)](https://www.mdpi.com/2673-9909/6/11/179/pdf)

> **Paper:** E. G. Sultanow, A. Hatziiliou, L. Faubel-Teich, M. Amir, *Hearing Ellipsoids: A Benchmark of Krylov, Spectral Galerkin, and Filter Diagonalization Eigensolvers*. AppliedMath 6 (2026), 179, [doi:10.3390/appliedmath6110179](https://doi.org/10.3390/appliedmath6110179). **Read it on [MDPI](https://www.mdpi.com/2673-9909/6/11/179).**

This repository contains the code, spectra, meshes, notebooks, and cluster jobs for
the paper **Hearing Ellipsoids**. The main benchmark compares structurally
different methods for computing large Dirichlet spectra on the triaxial ellipsoid

$$
\Omega(1,1.5,2.3)
=\left\lbrace (x,y,z):x^2+\frac{y^2}{1.5^2}+\frac{z^2}{2.3^2}<1\right\rbrace .
$$

The workflow covers FEM-ARPACK, SLEPc spectrum slicing, Stretched Spectral Galerkin
(SSG), and the gap-free resolvent block-Hankel method. Completeness of the spectra is
checked independently by Sylvester inertia counts.

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

The principal triaxial spectra reported in the paper are committed to the repository:

| Method | Eigenvalue file | Number of values |
|---|---|---:|
| FEM-ARPACK | [`ellipsoid_a1_b1.5_c2.3_P2_h0.06_arnoldi_highacc_N2000_eigs.txt`](data/ellipsoid-benchmark/outputs/arnoldi_reference_triaxial/ellipsoid_a1_b1.5_c2.3_P2_h0.06_arnoldi_highacc_N2000_eigs.txt) | 2000 |
| SSG | [`ellipsoid_a1_b1.5_c2.3_l36_n20_qr64_qt56_qp112_ssg_N2000_eigs.txt`](data/ellipsoid-benchmark/outputs/ssg_reference_triaxial/ellipsoid_a1_b1.5_c2.3_l36_n20_qr64_qt56_qp112_ssg_N2000_eigs.txt) | 2000 |
| Certified block-Hankel | [`full_spectrum_eigs_certified.txt`](data/ellipsoid-benchmark/outputs/hankel_full_spectrum_triaxial/full_spectrum_eigs_certified.txt) | 2034 |
| SLEPc spectrum slicing, $h=0.06$ | [`a1_b1.5_c2.3_P2_h0.06_top449.118296481_slepc_slicing_eigs.txt`](data/ellipsoid-benchmark/outputs/slepc_slicing_triaxial/a1_b1.5_c2.3_P2_h0.06_top449.118296481_slepc_slicing_eigs.txt) | 2034 |
| SLEPc spectrum slicing, $h=0.03$ | [`a1_b1.5_c2.3_P2_h0.03_top449.118296481_slepc_slicing_n32p2_eigs.txt`](data/ellipsoid-benchmark/outputs/slepc_slicing_triaxial_h003_i64/a1_b1.5_c2.3_P2_h0.03_top449.118296481_slepc_slicing_n32p2_eigs.txt) | 2050 |

The FEM-ARPACK, block-Hankel, inertia, and $h=0.06$ SLEPc runs use the same quadratic
FEM pencil generated from
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
environment rather than installed through ordinary PyPI wheels. For the refined mesh
($h=0.03$), PETSc and SLEPc must be built with 64-bit indices (paper, Section 3.4).

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

$$
N(\lambda)=A_0\lambda^{3/2}+A_1\lambda+A_2\lambda^{1/2},
\qquad
V=6\pi^2A_0,\quad S=-16\pi A_1,\quad C=6\pi^2A_2.
$$

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
| Curvature-focused $A_2/C$ validation | 31 | 524 |
| Equal-weight reconstruction of $V,S,C$ | 32 | 720 |

The complete methodology, equations, audit values, method-specific optima, and a
paper-ready methods paragraph are in
[`ellipsoid-benchmark/FIT_WINDOW_OPTIMIZATION.md`](ellipsoid-benchmark/FIT_WINDOW_OPTIMIZATION.md).
The corresponding notebook is
[`reverse_geometry_A2_validation.ipynb`](ellipsoid-benchmark/notebooks/reverse_geometry_A2_validation.ipynb).

## SLEPc reference runs

At $h=0.06$ the SLEPc reference is **serial at the MPI level** because the SciPy CSR
matrices are transferred directly to PETSc on one rank; do not start these runners with
`mpiexec`. The refined-mesh jobs at $h=0.03$ are the exception: they run under `mpirun`
on 32 ranks in two spectrum partitions.

Minimal triaxial smoke test, run from a clone of this repository:

```bash
cd ellipsoid-benchmark
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python scripts/run_slepc_slicing.py \
  --data-root ../data/ellipsoid-benchmark \
  --out-dir ../data/ellipsoid-benchmark/outputs/slepc_smoke_triaxial \
  --a 1 --b 1.5 --c 2.3 --mesh-h 0.06 --order 2 \
  --lambda-hi 100 \
  -st_type sinvert \
  -st_ksp_type preonly \
  -st_pc_type cholesky \
  -st_pc_factor_mat_solver_type mumps
```

Production jobs:

```bash
sbatch jobs/run_slepc_slicing.slurm                 # triaxial, h = 0.06
sbatch jobs/run_slepc_slicing_unitball.slurm        # unit ball, h = 0.06
sbatch jobs/run_slepc_slicing_h003_i64.slurm        # triaxial, h = 0.03, 64-bit PETSc
sbatch jobs/run_slepc_slicing_unitball_h003.slurm   # unit ball, h = 0.03, 64-bit PETSc
```

The Slurm jobs contain the paths of the cluster on which the paper runs were executed;
adapt the log directory, Conda environment, and repository path before submitting.

The mandatory internal acceptance check is `count_matches_inertia = true`. The level
count is a result, not a prescribed target: the paper run at $h=0.06$ returned 2034
levels, the run at $h=0.03$ returned 2050, both confirmed by inertia. See
[`ellipsoid-benchmark/SLEPC_TESTING.md`](ellipsoid-benchmark/SLEPC_TESTING.md) for the
full checklist.

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
wall times are compared; memory-bandwidth contention would invalidate the runtime
comparison.

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

If you use this repository, please cite:

```bibtex
@article{sultanow2026hearing,
  title   = {Hearing Ellipsoids: A Benchmark of Krylov, Spectral Galerkin,
             and Filter Diagonalization Eigensolvers},
  author  = {Sultanow, Eldar Gunter and Hatziiliou, Andreas and
             Faubel-Teich, Leonhard and Amir, Malik},
  journal = {AppliedMath},
  volume  = {6},
  number  = {11},
  pages   = {179},
  year    = {2026},
  doi     = {10.3390/appliedmath6110179},
  url     = {https://www.mdpi.com/2673-9909/6/11/179}
}
```
