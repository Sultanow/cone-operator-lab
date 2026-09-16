# Independent hyperbolic validation suite

This directory is intentionally separate from `bm_cusps/`.
It provides independent reference geometries and exploratory comparison models
for validating the hyperbolic FEM ingredients used in the Brooks--Makover work.

## What belongs here

### Core validation

- `hyperbolic_disk.py`
  - Dirichlet spectrum of a hyperbolic geodesic disk.
  - Compares the 2D Poincare-disk FEM against an independent radial
    Sturm--Liouville discretization.

- `bolza_closed.py`
  - Closed genus-2 surface from opposite-side identifications of a regular
    hyperbolic octagon.
  - Checks side pairings, one vertex class, the zero mode, and compares the
    first non-zero eigenvalue against the Bolza reference value.

- `pants_atom.py`
  - Right-angled hexagon / pair-of-pants length identities and holonomy closure.

- `hexagon_cell.py`, `pants.py`, `glue.py`, `genus_n.py`
  - Independent Fenchel--Nielsen-style construction route:
    right-angled hexagon -> pair of pants -> closed surface.
  - Provides Gauss--Bonnet/area and zero-mode checks at increasing topological
    complexity.

These files should remain sufficiently independent of `bm_cusps` that they can
catch shared-model or shared-implementation errors rather than merely repeating
production code.

### Exploratory only

`exploratory/stats_m1.py` and `exploratory/stats_goe.py` are NOT validation
certificates and are NOT Weil--Petersson sampling.

- `stats_m1.py` now uses `vals[1]` as `lambda_1`.  The previous prototype used
  the mean of `vals[1:4]`, which is only meaningful for a known threefold
  cluster and is not the first non-zero eigenvalue of a generic surface.
- `stats_goe.py` performs only a screening-grade nearest-neighbour-spacing
  experiment.  It uses the leading Weyl unfolding `N(lambda) ~ lambda` for a
  genus-2 surface of area `4*pi`; this is not a precision low-spectrum
  unfolding and must not be presented as a universality test.

## Quick checks

From the repository root:

```bash
python validation/selftest.py
```

Individual benchmarks:

```bash
python validation/hyperbolic_disk.py
python validation/bolza_closed.py
python validation/pants_atom.py
python validation/pants.py
python validation/genus_n.py
```

Exploratory M1 run:

```bash
python -m validation.exploratory.stats_m1 \
  --samples 120 \
  --out-dir validation_output
```

Exploratory spacing screen:

```bash
python -m validation.exploratory.stats_goe \
  --surfaces 45 \
  --out-dir validation_output
```

## Interpretation

Passing these tests validates selected numerical building blocks and known
reference cases.  It does **not** prove that a Brooks--Makover computation is
correct, nor does the Fenchel--Nielsen generator sample the Weil--Petersson
measure.
