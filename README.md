# cone-operator-lab

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23423742.svg)](https://doi.org/10.5281/zenodo.23423742)

Code, data and notebooks for a series of studies in spectral geometry: how much of the shape of a domain can be recovered from its Laplace eigenvalues, and how reliable are the large numerical spectra this requires? Further folders contain experiments on hyperbolic surfaces. Every project lives in its own folder with its own README, paper reference and instructions.

## Projects

| folder | topic | paper |
|--------|-------|-------|
| [**ellipsoid-benchmark**](ellipsoid-benchmark) | Benchmark of eigensolvers for large Dirichlet spectra on a triaxial ellipsoid (FEM–ARPACK, SLEPc spectrum slicing, stretched spectral Galerkin, gap-free block-Hankel); Weyl-based reverse geometry and inertia certification | *Hearing Ellipsoids*, AppliedMath 6 (2026), 179 |
| [**rectangle-and-box**](rectangle-and-box) | Side lengths of a planar rectangle and of a 3D box from their Dirichlet eigenvalues: Fourier analysis of the spectral fluctuations, heat-trace fit and cubic reconstruction | *Hearing the Sides*, arXiv:2511.23047 (2025); *Hearing the Edges*, Axioms 15 (2026), 284 |
| [**conformal-twin-lab-ipc**](conformal-twin-lab-ipc) | Mizhaev's genus-three polyhedron and its circle-packing hyperbolic twin: certified radii, discretely conformal embedding with IPC contact handling, Alexandrov-type search | *Reversible Hyperbolization of a Genus Three Polyhedron* (manuscript in `paper/`) |
| [**brooks-makover**](brooks-makover) | Random Brooks–Makover surfaces: spectral gap of the cusped surface versus its compactification, conformal factor near the cusps, independent hyperbolic validation suite (documentation in `bm_cusps/README.md` and `validation/README.md`) | research code |

The three spectral-geometry papers form one line of work. The rectangle and the box have spectra in closed form, so the inverse method can be tested without any eigensolver error. The triaxial ellipsoid has no closed-form ordered spectrum; its spectra have to be computed, and their completeness has to be certified.

## Publications

| year | paper | folder |
|------|-------|--------|
| 2026 | E. G. Sultanow, A. Hatziiliou, L. Faubel-Teich, M. Amir: *Hearing Ellipsoids: A Benchmark of Krylov, Spectral Galerkin, and Filter Diagonalization Eigensolvers*. AppliedMath 6 (2026), 179, [doi:10.3390/appliedmath6110179](https://doi.org/10.3390/appliedmath6110179) | [ellipsoid-benchmark](ellipsoid-benchmark) |
| 2026 | E. Sultanow, A. Hatziiliou, C. May, N. Kratzke: *Hearing the Edges: Recovering a 3D Rectangular Box from Dirichlet Eigenvalues*. Axioms 15 (2026), 284, [doi:10.3390/axioms15040284](https://doi.org/10.3390/axioms15040284) | [rectangle-and-box](rectangle-and-box) |
| 2025 | E. Sultanow, A. Hatziiliou: *Hearing the Sides: Recovering a Planar Rectangle from Eigenvalues*. Preprint, [arXiv:2511.23047](https://arxiv.org/abs/2511.23047) | [rectangle-and-box](rectangle-and-box) |

BibTeX entries are in the README of each folder.

## Repository layout

```text
cone-operator-lab/
├── ellipsoid-benchmark/          # Hearing Ellipsoids: Python package, Slurm jobs, scripts, notebooks
├── rectangle-and-box/            # Hearing the Sides and Hearing the Edges: notebooks, modules, data
├── conformal-twin-lab-ipc/       # genus-three polyhedron and its hyperbolic twin
├── brooks-makover/               # Brooks–Makover surfaces, cusps and compactification
├── data/
│   ├── ellipsoid-benchmark/      # meshes, spectra, certificates and tables of the ellipsoid paper
│   └── eigenvalues/              # earlier ellipsoid spectra and the unit-ball reference
├── mathematica/                  # earlier ellipsoid notebooks, unit-ball eigenvalues, Render_Ellipsoid.nb
├── python/                       # earlier ellipsoid solvers and notebooks
├── plots/                        # rendering of the working ellipsoid
└── deploy-ellipsoid-benchmark.sh # copies the benchmark code to the cluster
```

`data/eigenvalues/`, `mathematica/`, `python/` and `plots/` hold the earlier ellipsoid computations that preceded the benchmark package: eigenvalue generators, Weyl and heat-trace reconstructions and the unit-ball spectrum. They stay at the top level because the ellipsoid paper links to [`mathematica/Render_Ellipsoid.nb`](mathematica/Render_Ellipsoid.nb) (Figure 1). [`python/notebooks/Asymptotic_Fingerprints.ipynb`](python/notebooks/Asymptotic_Fingerprints.ipynb) is a prototype on asymptotic fingerprints of cone differential operators.

## Tags

- [`v1.0`](https://github.com/Sultanow/cone-operator-lab/tree/v1.0) – release archived on Zenodo, [doi:10.5281/zenodo.23423742](https://doi.org/10.5281/zenodo.23423742).
- [`layout-as-cited`](https://github.com/Sultanow/cone-operator-lab/tree/layout-as-cited) – the layout before the rectangle and box code moved into its own folder; the file paths printed in *Hearing the Sides* and *Hearing the Edges* resolve there.
