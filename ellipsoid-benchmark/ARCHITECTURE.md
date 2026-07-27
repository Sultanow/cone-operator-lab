# Code Architecture

## Goal

The package separates four concerns:

1. **Problem construction**  
   Geometry, meshing, FEM assembly.

2. **Algorithms**  
   Solvers expose one clean contract: input `K, M`; output `AlgorithmResult`.

3. **Validation**  
   Unit-ball truth, cluster/windows, Weyl reverse geometry, DMC ground-state checks.

4. **Orchestration**  
   Thin benchmark scripts and notebooks call reusable modules.

## Package layout

```text
hearing_ellipsoid_bench/
  core/
    types.py                # AlgorithmResult, configs, cleaning
  io/
    eigenvalues.py          # load/save eigenvalue lists
  geometry/
    ellipsoid.py            # volume, surface area
  fem/
    assembly.py             # gmsh -> meshio -> scikit-fem K,M
  solvers/
    arpack.py               # shift-invert and window slicing
    slepc.py                # SLEPc/PETSc Krylov-Schur
    fdm_block.py            # multi-probe Block-Hankel FDM
    ssg.py                  # Stretched Spectral Galerkin extraction
  validation/
    sphere.py               # unit-ball comparison, bands, clusters
    weyl.py                 # reverse geometry via Weyl
  dmc/
    validation.py           # bridge-corrected DMC lambda_1
  benchmarks/
    unit_ball.py            # a=b=c=1 benchmark
    reverse_geometry.py     # a=1,b=1.5,c=2.3 Weyl loops
  viz/
    plots.py
```

## Algorithm contract

Every solver returns:

```python
AlgorithmResult(
    algorithm="slepc_krylov_schur_lowest",
    eigs=np.ndarray,
    runtime_sec=float,
    success=True,
    message="",
    meta={...}
)
```

That lets the benchmark layer treat ARPACK, SLEPc, FDM, SSG and future methods uniformly.

## Benchmark tracks

### Track A: Unit ball

```python
df = compare_to_truth(eigs_num, eigs_true, n=5000)
bands = error_bands(df)
```

This answers: *How far is the algorithm from known sphere values?*

### Track B: Reverse geometry

```python
df = reverse_geometry_table(
    {"SSG": ssg_eigs, "Arnoldi": arnoldi_eigs},
    a=1.0, b=1.5, c=2.3,
    k_values=[100, 200, 300, 500, 1000]
)
```

This answers: *How well does the spectrum reconstruct volume/surface via Weyl?*

## Notebook rule

Notebooks should contain:

- parameter cells,
- one or two function calls,
- plots,
- discussion.

They should not contain algorithm definitions anymore.
