# Hearing Ellipsoid Bench

Clean Python package for benchmarking algorithms that generate large Dirichlet eigenvalue spectra on triaxial ellipsoids.

## Two validation tracks

1. **Unit-ball ground truth**  
   Set `a=b=c=1` and compare numerical spectra against precomputed unit-ball eigenvalues.

2. **Reverse geometry / Weyl validation**  
   For a real triaxial ellipsoid such as `a=1, b=1.5, c=2.3`, estimate volume and surface-related quantities via Weyl asymptotics over growing prefix sizes `K = 100, 200, 300, 500, 1000, ...`.

## Algorithm families

- FEM + ARPACK
- FEM + SLEPc/PETSc
- Windowed spectral slicing
- Scalar FDM / filter diagonalization
- Multi-probe Block-Hankel FDM
- Stretched Spectral Galerkin (SSG)
- DMC validation for the ground-state eigenvalue

## Quick start

```bash
cd /home/jovyan/hearing-ellipsoid-bench
pip install -e .
```

```python
from hearing_ellipsoid_bench.io.eigenvalues import load_eigenvalues_txt
from hearing_ellipsoid_bench.validation.sphere import compare_to_truth
```

The notebooks should become thin orchestration layers. The reusable logic lives in `src/hearing_ellipsoid_bench`.

## Reverse-geometry fit window

The exhaustive stable selection of `FIT_START` and `FIT_END_MAX` is documented in
[`FIT_WINDOW_OPTIMIZATION.md`](FIT_WINDOW_OPTIMIZATION.md). Reproduce it with:

```bash
python scripts/optimize_reverse_geometry_window.py --dry-run
```
