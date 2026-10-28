# Hearing Ellipsoids – benchmark

[![AppliedMath](https://img.shields.io/badge/AppliedMath-6%2811%29%2C%20179%20%282026%29-orange)](https://www.mdpi.com/2673-9909/6/11/179) [![DOI](https://img.shields.io/badge/DOI-10.3390%2Fappliedmath6110179-blue)](https://doi.org/10.3390/appliedmath6110179) [![Open Access](https://img.shields.io/badge/Paper-open%20access-lightgrey)](https://www.mdpi.com/2673-9909/6/11/179/pdf)

> **Paper:** E. G. Sultanow, A. Hatziiliou, L. Faubel-Teich, M. Amir, *Hearing Ellipsoids: A Benchmark of Krylov, Spectral Galerkin, and Filter Diagonalization Eigensolvers*. AppliedMath 6 (2026), 179, [doi:10.3390/appliedmath6110179](https://doi.org/10.3390/appliedmath6110179). **Read it on [MDPI](https://www.mdpi.com/2673-9909/6/11/179).** This folder contains the solvers, Slurm jobs, scripts and notebooks behind all numerical results of the paper; the meshes, spectra, certification reports and result tables are in [`data/ellipsoid-benchmark`](../data/ellipsoid-benchmark).

The benchmark compares structurally different methods for computing large Dirichlet spectra on the triaxial ellipsoid

$$
\Omega(1,1.5,2.3)
=\left\lbrace (x,y,z):x^2+\frac{y^2}{1.5^2}+\frac{z^2}{2.3^2}<1\right\rbrace
$$

and on the unit ball. Three unrelated solvers on the same finite-element matrices (FEM–ARPACK, SLEPc spectrum slicing, gap-free block-Hankel) agree to a median relative deviation of $3\times10^{-11}$, and Sylvester inertia counts certify the level count exactly at about 0.05 % of the cost of the spectrum. For this problem size, spectrum slicing is by far the cheapest route (4.9 core-hours against roughly 69,400 for the block-Hankel run). As the eigenvalue index grows, the error in the reconstructed geometry is dominated by the mesh, not by the eigensolver: SSG, which needs no volume mesh, reconstructs the volume to about 0.025 %, and halving the mesh size to $h=0.03$ improves the curvature term of the finite-element spectrum by a factor of 31.

The reusable logic is the Python package `hearing-ellipsoid-bench` in `src/hearing_ellipsoid_bench`; scripts and notebooks are thin orchestration layers on top of it.

The benchmark builds on two predecessor studies on domains with closed-form spectra, the planar rectangle and the 3D box, see [`../rectangle-and-box`](../rectangle-and-box).

## Two validation tracks

1. **Unit-ball ground truth**  
   Set `a=b=c=1` and compare numerical spectra against the exact unit-ball eigenvalues $\lambda_{\ell,k}=j_{\ell+1/2,k}^2$ with multiplicity $2\ell+1$ ([`unit_ball_N20000.txt`](../data/ellipsoid-benchmark/reference/unit_ball_N20000.txt), first 20,000 levels).

2. **Reverse geometry / Weyl validation**  
   For a real triaxial ellipsoid such as `a=1, b=1.5, c=2.3`, fit the three-term Weyl law to growing spectral prefixes `K = 100, 200, 300, 500, 1000, ...` and compare the recovered volume, surface area and integrated mean curvature with their values from the semi-axes.

## Algorithm families

Evaluated in the paper:

| family | solver code | paper |
|--------|-------------|-------|
| FEM + ARPACK (shift-invert Arnoldi) | `solvers/arpack.py`, `solvers/arnoldi_reference.py` | Sections 2.1, 3.2, 5.1 |
| FEM + SLEPc/PETSc (Krylov–Schur spectrum slicing, MUMPS inertia) | `solvers/slepc.py` | Sections 2.4, 3.4, 5.3 |
| Stretched Spectral Galerkin (SSG) | `solvers/ssg.py` | Sections 2.2, 3.3, 5.2 |
| Multi-probe Block-Hankel FDM, gap-free resolvent variant | `solvers/fdm_block.py`, `scripts/run_hankel_band.py` | Sections 2.3, 3.5, 5.4 |
| Inertia certification (solver-agnostic) | `scripts/certify_inertia.py` | Section 3.6, Appendix D |

The package also contains code outside the scope of the paper: windowed spectral slicing, scalar FDM / filter diagonalization, a boundary method of fundamental solutions (`solvers/boundary_mfs.py`) and DMC validation for the ground-state eigenvalue (`dmc/`).

## Layout

```text
ellipsoid-benchmark/
├── src/hearing_ellipsoid_bench/  # reusable Python package
├── scripts/                      # local and post-processing runners
├── jobs/                         # Slurm jobs
├── notebooks/                    # validation and plotting notebooks
├── examples/                     # minimal usage examples
├── tests/                        # lightweight regression tests
├── ARCHITECTURE.md               # code architecture
├── SLEPC_TESTING.md              # SLEPc smoke test and paper runs
└── FIT_WINDOW_OPTIMIZATION.md    # Weyl fit-window methodology

data/ellipsoid-benchmark/         # at the repository root
├── *.msh                         # shared FEM meshes
├── reference/                    # exact unit-ball data
├── generated/                    # generated validation spectra
└── outputs/                      # paper spectra, certificates and result tables
```

## Installation and tests

Python 3.10 or newer is required.

```bash
git clone https://github.com/Sultanow/cone-operator-lab.git
cd cone-operator-lab/ellipsoid-benchmark
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

The tests do not require a cluster or a working SLEPc installation; the SLEPc configuration test uses a controlled mock backend.

```python
from hearing_ellipsoid_bench.io.eigenvalues import load_eigenvalues_txt
from hearing_ellipsoid_bench.validation.sphere import compare_to_truth
```

PETSc, SLEPc, MUMPS and their Python bindings (`pip install -e ".[slepc]"`) are needed only for the SLEPc runs. On a cluster they should come from the existing module or Conda environment rather than from ordinary PyPI wheels. For the refined mesh ($h=0.03$), PETSc and SLEPc must be built with 64-bit indices; the conda-forge packages use 32-bit indices and end in a segmentation fault after about ninety minutes instead of a clear error message (paper, Section 3.4 and Listing 2).

## Data availability

The principal triaxial spectra reported in the paper:

| Method | Eigenvalue file | Number of values |
|---|---|---:|
| FEM-ARPACK | [`ellipsoid_a1_b1.5_c2.3_P2_h0.06_arnoldi_highacc_N2000_eigs.txt`](../data/ellipsoid-benchmark/outputs/arnoldi_reference_triaxial/ellipsoid_a1_b1.5_c2.3_P2_h0.06_arnoldi_highacc_N2000_eigs.txt) | 2000 |
| SSG | [`ellipsoid_a1_b1.5_c2.3_l36_n20_qr64_qt56_qp112_ssg_N2000_eigs.txt`](../data/ellipsoid-benchmark/outputs/ssg_reference_triaxial/ellipsoid_a1_b1.5_c2.3_l36_n20_qr64_qt56_qp112_ssg_N2000_eigs.txt) | 2000 |
| Certified block-Hankel | [`full_spectrum_eigs_certified.txt`](../data/ellipsoid-benchmark/outputs/hankel_full_spectrum_triaxial/full_spectrum_eigs_certified.txt) | 2034 |
| SLEPc spectrum slicing, $h=0.06$ | [`a1_b1.5_c2.3_P2_h0.06_top449.118296481_slepc_slicing_eigs.txt`](../data/ellipsoid-benchmark/outputs/slepc_slicing_triaxial/a1_b1.5_c2.3_P2_h0.06_top449.118296481_slepc_slicing_eigs.txt) | 2034 |
| SLEPc spectrum slicing, $h=0.03$ | [`a1_b1.5_c2.3_P2_h0.03_top449.118296481_slepc_slicing_n32p2_eigs.txt`](../data/ellipsoid-benchmark/outputs/slepc_slicing_triaxial_h003_i64/a1_b1.5_c2.3_P2_h0.03_top449.118296481_slepc_slicing_n32p2_eigs.txt) | 2050 |

The FEM-ARPACK, block-Hankel, inertia and $h=0.06$ SLEPc runs use the same quadratic FEM pencil generated from [`ellipsoid_a1_b1.5_c2.3_h0.06.msh`](../data/ellipsoid-benchmark/ellipsoid_a1_b1.5_c2.3_h0.06.msh), so the stiffness and mass matrices are identical across solvers. SSG is independent of the volume mesh.

## Reproducing the paper

All production runs were executed as Slurm batch jobs on the Research Computing Infrastructure of the EDIH Hamburg project. Results are written to [`data/ellipsoid-benchmark/outputs`](../data/ellipsoid-benchmark/outputs). The Slurm jobs contain the paths of that cluster; adapt the log directory, Conda environment, repository path and `--data-root` before submitting.

| result | script | Slurm job |
|--------|--------|-----------|
| FEM–ARPACK, triaxial and unit ball | `scripts/run_arnoldi_reference.py` | `jobs/run_arnoldi_h006_N2000_ncv20*.slurm` |
| SSG, triaxial and unit ball | `scripts/run_ssg_reference.py` | `jobs/run_ssg_l36_n20_q64_56_112_N2000_mem700.slurm`, `jobs/run_ssg_unitball_l36_n20_N2000.slurm` |
| SLEPc spectrum slicing, $h=0.06$ and $h=0.03$ | `scripts/run_slepc_slicing.py`, `scripts/run_slepc_slicing_unitball.py` | `jobs/run_slepc_slicing*.slurm` |
| Gap-free block-Hankel, 48 (band, seed) tasks | `scripts/run_hankel_band.py` | `jobs/run_hankel_full_spectrum_array*.slurm`, chained by `jobs/submit_chain.sh` |
| Consensus merge (Appendix C) | `scripts/merge_hankel_full_spectrum.py` | part of the array job |
| Rank-cap test on the unit ball (Section 5.4) | `scripts/run_hankel_band.py` | `jobs/run_ballrank_test.slurm` |
| Inertia certification (Section 3.6, Appendix D) | `scripts/certify_inertia.py` | `jobs/run_certify_inertia*.slurm` |
| Cross-method comparison (Appendix E) | `scripts/compare_vs_reference.py`, `scripts/adjudicate_positions.py` | – |
| Multiplicities against the ball spectrum (Appendix F) | `scripts/validate_unitball_multiplicities.py` | – |
| Figure 1 (working ellipsoid) | [`mathematica/Render_Ellipsoid.nb`](../mathematica/Render_Ellipsoid.nb) | – |
| Figure 2 (reverse geometry) | [`notebooks/reverse_geometry_A2_validation.ipynb`](notebooks/reverse_geometry_A2_validation.ipynb) | – |
| Figure 3 (unit-ball deviation) | [`notebooks/unit_ball_ground_truth_validation.ipynb`](notebooks/unit_ball_ground_truth_validation.ipynb) | – |

Do not run production SLEPc and block-Hankel jobs concurrently on the same nodes when wall times are compared; memory-bandwidth contention would invalidate the runtime comparison.

## SLEPc reference runs

At $h=0.06$ the SLEPc reference is **serial at the MPI level** because the SciPy CSR matrices are transferred directly to PETSc on one rank; do not start these runners with `mpiexec`. The refined-mesh jobs at $h=0.03$ are the exception: they run under `mpirun` on 32 ranks in two spectrum partitions.

Minimal triaxial smoke test:

```bash
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

The mandatory internal acceptance check is `count_matches_inertia = true`. The level count is a result, not a prescribed target: the paper run at $h=0.06$ returned 2034 levels, the run at $h=0.03$ returned 2050, both confirmed by inertia. See [`SLEPC_TESTING.md`](SLEPC_TESTING.md) for the full checklist.

## Stable Weyl fit-window selection

The reverse-geometry notebook fits

$$
N(\lambda)=A_0\lambda^{3/2}+A_1\lambda+A_2\lambda^{1/2},
\qquad
V=6\pi^2A_0,\quad S=-16\pi A_1,\quad C=6\pi^2A_2 .
$$

An exhaustive laptop-scale search evaluates every admissible integer fit window and a nine-point neighborhood around it. The loss combines the neighborhood median error, the interquartile range of the curvature error, and the worst case over FEM-ARPACK, SSG, and block-Hankel.

```bash
python scripts/optimize_reverse_geometry_window.py --dry-run
```

With the committed spectra, the selected common windows are:

| Objective | `FIT_START` | `FIT_END_MAX` |
|---|---:|---:|
| Curvature-focused $A_2/C$ validation | 31 | 524 |
| Equal-weight reconstruction of $V,S,C$ | 32 | 720 |

The complete methodology, equations, audit values, method-specific optima, and a paper-ready methods paragraph are in [`FIT_WINDOW_OPTIMIZATION.md`](FIT_WINDOW_OPTIMIZATION.md); the corresponding notebook is [`reverse_geometry_A2_validation.ipynb`](notebooks/reverse_geometry_A2_validation.ipynb).

## Reproducibility notes

- Preserve the committed meshes and use `force_remesh=False` for cross-solver runs.
- Compare spectra at a common spectral cutoff, not merely at the same list length.
- Retain Slurm logs and `sacct` resource reports with every production result.
- Treat a fit-window optimum as conditional on its declared loss and admissible set.
- Do not describe a spectrum as certified unless the corresponding inertia checks pass.

## Further documentation

- [Code architecture](ARCHITECTURE.md)
- [SLEPc testing and paper runs](SLEPC_TESTING.md)
- [Stable fit-window optimization](FIT_WINDOW_OPTIMIZATION.md)

## Citation

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

## Acknowledgments

The production runs were executed on the Research Computing Infrastructure of the EDIH Hamburg project at the University of Hamburg. We thank Kai Himstedt and Stephanie von Riegen for access and support, and Elmar Schrohe for suggesting the study of inverse spectral questions.
