# 3-D print: Brooks–Makover compactification

`bmprint3d.py` creates a **watertight STL/OBJ sculpture** for one deterministic
Brooks–Makover sample `(n, seed)`.

## What the print means

The FEM code stores the hyperbolic surface intrinsically, chart by chart.  A
closed hyperbolic surface of genus `g >= 2` does **not** admit a smooth global
isometric embedding in ordinary Euclidean 3-space.  The print is therefore a
**topologically faithful Euclidean realization of the compactified surface**
`Sbar`, not a claim that Euclidean distances on the STL equal hyperbolic
distances.

The solid is the regular neighbourhood of a graph with `g` loops, so its
boundary has genus `g`.  The script validates this from the Euler
characteristic of the final triangle mesh before accepting the export.

Optional tactile bumps encode the BM cusp-cycle lengths:

- short cusp cycle `k` -> larger bump,
- long cusp cycle `k` -> smaller bump.

These bumps are **annotations** of the original cusped surface.  In `Sbar` the
punctures themselves have been filled.

## Install

```bash
pip install -r requirements.txt
```

## Recommended first model

The included experiments show that `n=16, seed=1` has genus 6 and cusp lengths
including `k=1` and `k=2`.  It is complex enough to be interesting but still
practical to print and hold in one hand.

Quick preview:

```bash
python bmprint3d.py --n 16 --seed 1 --size-mm 160 --resolution 4 --preview
```

Final mesh:

```bash
python bmprint3d.py --n 16 --seed 1 --size-mm 160 --resolution 7 --preview \
  --out prints/bm_n16_s1
```

Outputs:

```text
prints/bm_n16_s1.stl       # slicer / 3-D printer
prints/bm_n16_s1.obj       # Blender / MeshLab
prints/bm_n16_s1.json      # topology + print validation report
prints/bm_n16_s1_preview.png
```

The command exits with an error if the mesh is not watertight or if the genus
computed from the STL differs from the BM genus.

## Slicer recommendation

For a first FDM test use PLA, 0.2 mm layer height, 3 walls and supports where
needed.  The JSON report includes an estimated structural tube diameter.  If it
is below about 2.2 mm, increase `--size-mm`.

The model is intentionally elongated because this keeps individual handles
separated and the topology reliable.  For very high genus (for example the
`n=128` example with genus 61), use a much larger print, lower preview
resolution, or choose a lower-genus representative sample for a hand-held
exhibit.

## Remove cusp markers

For a pure genus-g surface:

```bash
python bmprint3d.py --n 16 --seed 1 --no-cusp-markers --preview
```

## Scientific caption

> Euclidean topological realization of the compactified Brooks–Makover surface
> `Sbar`.  The printed embedding preserves the topology (genus) but is not an
> isometric embedding of the intrinsic hyperbolic metric.  Tactile bumps encode
> cusp-cycle lengths of the pre-compactification surface; shorter cycles are
> represented by larger markers.
