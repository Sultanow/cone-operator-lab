# Hearing Ellipsoids – issue cover

Cover image for **Hearing Ellipsoids: A Benchmark of Krylov, Spectral Galerkin, and Filter Diagonalization Eigensolvers**, *AppliedMath* 2026, 6(11), 179, [doi:10.3390/appliedmath6110179](https://doi.org/10.3390/appliedmath6110179).

The image is computed, not drawn: every pixel follows deterministically from a finite-difference eigenvalue computation and a ray caster. No generative image model is involved.

## What the image shows

| Element | Content |
| --- | --- |
| Body | The paper’s working ellipsoid Ω(1, 1.5, 2.3), cut away along its three symmetry planes |
| Cut faces | A Dirichlet eigenfunction *u* from the fully even symmetry class, λ ≈ 80.6 on the *h* = 0.03 grid |
| Outer shell | The boundary flux ∂*u*/∂*n* of the same eigenfunction |
| Rings | The lowest 123 eigenvalues (all 8 symmetry classes, *h* = 0.035), placed at their wave numbers √λ<sub>k</sub>; brighter rings mark near-coinciding levels |

The rings use coarse finite-difference values (about 2–3 % low because of the staircase boundary) for illustration. They are **not** the certified 2034-level spectrum from the paper. To use the certified data instead, pass a sorted `.npy` array of eigenvalues via `--spec`.

## Files

```
code/
  solver.py        FD Dirichlet Laplacian on one octant, per parity class (x, y, z) ∈ {±1}³
  scan.py          30 fully even modes near λ = 60, plus slice thumbnails
  spec_class.py    low spectrum per symmetry class, merged into fd_spectrum.npy
  render.py        ray-cast cut-away, boundary flux, spectral rings, compositing
  fd_spectrum.npy  the 123 levels used for the rings
layers/
  HearingEllipsoids_cover_layer1_background.png      opaque gradient
  HearingEllipsoids_cover_layer2_spectral_rings.png  RGBA, transparent
  HearingEllipsoids_cover_layer3_ellipsoid.png       RGBA, transparent
```

All layers are 2480 × 3508 px at 300 dpi (A4 portrait). The soft blue glow behind the ellipsoid exists only in the composite, not in the layers.

## Reproduce

Requirements: Python 3, NumPy, SciPy, Pillow (Matplotlib only for the thumbnails in `scan.py`). The whole pipeline runs in about two minutes on a single core.

```bash
cd code

# 1. Fully even eigenmodes near lambda = 60  ->  scan_even.npz, scan_even.png
python scan.py

# 2. Low spectrum for all 8 symmetry classes  ->  spec_c*.npy, merged into fd_spectrum.npy
python spec_class.py 0 1 2 3 4 5 6 7

# 3. Render the ellipsoid at 2x supersampling in four row chunks
for i in 0 1 2 3; do
  python render.py --mode 21 --W 2480 --H 3508 --ss 2 --part $i/4
done

# 4. Composite and export the layers
python render.py --mode 21 --W 2480 --H 3508 --assemble 4 --layers --out cover.png
```

For a quick low-resolution preview, use `python render.py --mode 21`. The default size is 620 × 877 px.

## Method notes

- **Discretisation:** cell-centred 7-point Laplacian on the octant *x, y, z* > 0. Points outside the ellipsoid are set to zero (Dirichlet, staircase boundary). On the symmetry planes, the ghost value is ±*u* according to the parity class. As a sanity check on the unit ball, the solver gives λ₁ = 9.69 at *h* = 0.03, against the exact value π² ≈ 9.87.
- **Eigensolver:** `scipy.sparse.linalg.eigsh` in shift-invert mode.
- **Field evaluation:** the octant solution is mirrored to the full box and sampled with cubic B-splines (`scipy.ndimage.map_coordinates`). The boundary flux is approximated as ∂*u*/∂*n* ≈ −*u*(*p* − δ*n*)/δ with δ = 0.06.
- **Rendering:** analytic ray–ellipsoid and ray–plane intersections with perspective projection, Phong shading with Fresnel rim, and 2× supersampling.
- **Rings:** concentric ellipses fitted to the silhouette’s second moments. Radii are linear in √λ<sub>k</sub>, so the ring density grows outward as the Weyl law predicts.
