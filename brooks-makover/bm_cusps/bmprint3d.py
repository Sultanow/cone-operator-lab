#!/usr/bin/env python3
"""bmprint3d.py -- printable 3-D sculpture of a Brooks--Makover compactification.

This module deliberately separates intrinsic geometry from a Euclidean display
embedding.  A compact hyperbolic surface of genus g>=2 has no smooth isometric
embedding in R^3.  The STL/OBJ produced here is therefore a *topologically
faithful tactile model* of the compactified surface Sbar, not an isometric
rendering of its hyperbolic metric.

Construction
------------
For a BM sample (n, seed) we rebuild the exact ribbon graph, read its genus g
and cusp lengths k_c, and create a genus-g handlebody as the regular
neighbourhood of a spine with g attached loops.  The boundary of this
handlebody is a closed orientable surface of genus g.  Cusp data are encoded by
small tactile bumps on the side opposite the handles; shorter cusp cycles make
larger bumps.  These markers are annotation only and do not represent literal
Euclidean cusp geometry.

The solid is generated as an implicit union and meshed with marching cubes,
which gives a closed printable triangle mesh.  Trimesh is then used for cleanup,
validation, scaling in millimetres and STL/OBJ export.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from bmsurf import BMSurface

try:
    from skimage import measure
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("bmprint3d requires scikit-image: pip install scikit-image") from exc

try:
    import trimesh
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("bmprint3d requires trimesh: pip install trimesh") from exc


def _segment_distance_grid(X, Y, Z, a, b):
    """Distance from all grid points to a 3-D line segment a--b."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ab = b - a
    den = float(np.dot(ab, ab))
    if den == 0:
        return np.sqrt((X-a[0])**2 + (Y-a[1])**2 + (Z-a[2])**2)
    t = ((X-a[0])*ab[0] + (Y-a[1])*ab[1] + (Z-a[2])*ab[2]) / den
    t = np.clip(t, 0.0, 1.0)
    dx = X - (a[0] + t*ab[0])
    dy = Y - (a[1] + t*ab[1])
    dz = Z - (a[2] + t*ab[2])
    return np.sqrt(dx*dx + dy*dy + dz*dz)


def _circle_distance_grid(X, Y, Z, x0, radius):
    """Distance to circle in the yz-plane, centred at (x0, radius, 0).

    The circle has radius ``radius`` and passes through (x0,0,0), i.e. it is
    attached to the x-axis spine at exactly one graph vertex before thickening.
    """
    radial = np.sqrt((Y - radius)**2 + Z**2)
    return np.sqrt((X - x0)**2 + (radial - radius)**2)


def _sphere_sdf(X, Y, Z, c, r):
    return np.sqrt((X-c[0])**2 + (Y-c[1])**2 + (Z-c[2])**2) - r


def _build_field(genus: int, cusp_lengths, pitch=3.0, loop_radius=1.15,
                 tube_radius=0.36, marker_scale=0.34, voxels_per_tube=5.5):
    """Return (field, axes, metadata) for the implicit printable solid."""
    if genus < 1:
        raise ValueError("genus must be >= 1")

    # Dimensionless geometry.  Adjacent graph vertices are pitch*tube_radius
    # apart, so handles remain separated under normal parameters.
    tr = float(tube_radius)
    spacing = float(pitch) * tr
    R = float(loop_radius)
    xs = (np.arange(genus) - 0.5*(genus-1)) * spacing
    x0 = xs[0] - 1.6*tr
    x1 = xs[-1] + 1.6*tr

    # Tactile cusp markers:  r(k) decreases gently with sqrt(k), clipped so a
    # very long cusp remains visible.  Markers sit on the -z side of the spine,
    # opposite the handle loops, and overlap the tube so the final solid stays
    # connected and watertight.
    cusp_lengths = [int(k) for k in cusp_lengths]
    marker_specs = []
    if cusp_lengths:
        mx = np.linspace(x0 + 0.5*tr, x1 - 0.5*tr, len(cusp_lengths))
        for xx, k in zip(mx, cusp_lengths):
            rel = max(0.28, 1.0 / math.sqrt(max(1, k)))
            rr = tr * (0.34 + marker_scale * rel)
            # overlap the spine by about 35% of marker radius
            cz = -(tr + 0.65*rr)
            marker_specs.append((float(xx), 0.0, float(cz), float(rr), int(k)))

    margin = 2.6*tr
    ymin = -margin
    ymax = 2.0*R + margin
    zext = R + tr + margin
    xmin = x0 - margin
    xmax = x1 + margin
    if marker_specs:
        zmin = min(-zext, min(cz-r-margin for _,_,cz,r,_ in marker_specs))
    else:
        zmin = -zext
    zmax = zext

    dx = tr / float(voxels_per_tube)
    nx = max(32, int(math.ceil((xmax-xmin)/dx))+1)
    ny = max(48, int(math.ceil((ymax-ymin)/dx))+1)
    nz = max(48, int(math.ceil((zmax-zmin)/dx))+1)

    # Limit accidental gigantic allocations.  The user can lower --resolution
    # (voxels_per_tube) for high genus models.
    nvox = nx*ny*nz
    if nvox > 160_000_000:
        raise MemoryError(
            f"implicit grid would contain {nvox/1e6:.0f} M voxels; "
            "lower --resolution or use a smaller-genus demonstration surface")

    xv = np.linspace(xmin, xmax, nx, dtype=np.float32)
    yv = np.linspace(ymin, ymax, ny, dtype=np.float32)
    zv = np.linspace(zmin, zmax, nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(xv, yv, zv, indexing="ij", sparse=True)

    # Regular neighbourhood of the spine segment.
    field = _segment_distance_grid(X, Y, Z, (x0,0,0), (x1,0,0)) - tr

    # g circles attached to the spine.  Thickening a graph with first Betti
    # number g yields a genus-g handlebody; its boundary is our closed model.
    for xx in xs:
        field = np.minimum(field, _circle_distance_grid(X, Y, Z, float(xx), R) - tr)

    # Cusp markers are annotation bumps and should not alter genus.
    for xx, yy, zz, rr, _k in marker_specs:
        field = np.minimum(field, _sphere_sdf(X, Y, Z, (xx,yy,zz), rr))

    meta = dict(
        grid=[int(nx), int(ny), int(nz)], voxel_size_dimensionless=float(dx),
        loop_centres_x=xs.tolist(), spine=[float(x0), float(x1)],
        marker_specs=[dict(x=x,y=y,z=z,r=r,k=k) for x,y,z,r,k in marker_specs],
    )
    return field.astype(np.float32, copy=False), (xv,yv,zv), meta


def printable_model(n: int, seed: int, size_mm: float = 160.0,
                    tube_mm_min: float = 2.2, resolution: float = 5.5,
                    include_markers: bool = True):
    """Build one deterministic BM sample and return (trimesh, report)."""
    rng = np.random.default_rng(seed)
    surf = BMSurface.random(n, rng)
    if surf.g < 2:
        raise ValueError(f"sample has genus {surf.g}; choose a BM sample with genus >= 2")

    ks = surf.k.tolist() if include_markers else []
    field, axes, meta = _build_field(surf.g, ks, voxels_per_tube=resolution)
    xv, yv, zv = axes
    spacing = (float(xv[1]-xv[0]), float(yv[1]-yv[0]), float(zv[1]-zv[0]))

    verts, faces, _normals, _values = measure.marching_cubes(field, level=0.0, spacing=spacing)
    verts[:, 0] += float(xv[0]); verts[:, 1] += float(yv[0]); verts[:, 2] += float(zv[0])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh, multibody=True)

    # Centre and scale to target x extent in millimetres.
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    extent_x = float(mesh.extents[0])
    scale = float(size_mm) / extent_x
    mesh.apply_scale(scale)

    # For ordinary FDM printing, warn if the structural tube diameter is too
    # thin after global scaling.  This does not change the geometry silently.
    tube_diameter_mm = 2.0 * 0.36 * scale

    # Topological sanity check from Euler characteristic if the mesh is closed.
    euler = int(mesh.euler_number)
    genus_mesh = None
    if mesh.is_watertight and euler % 2 == 0:
        genus_mesh = int((2 - euler)//2)

    report = dict(
        generator="bmprint3d",
        interpretation=("topologically faithful Euclidean sculpture of Sbar; "
                        "NOT an isometric embedding of the hyperbolic metric"),
        n=int(n), seed=int(seed), attempts=int(getattr(surf, "attempts", 1)),
        triangles=int(2*n), genus_expected=int(surf.g), cusps=int(surf.V),
        cusp_lengths=sorted([int(k) for k in surf.k], reverse=True),
        mesh_vertices=int(len(mesh.vertices)), mesh_faces=int(len(mesh.faces)),
        watertight=bool(mesh.is_watertight), winding_consistent=bool(mesh.is_winding_consistent),
        euler_number=euler, genus_mesh=genus_mesh,
        extents_mm=[float(x) for x in mesh.extents],
        tube_diameter_mm_estimate=float(tube_diameter_mm),
        minimum_recommended_tube_mm=float(tube_mm_min),
        thin_warning=bool(tube_diameter_mm < tube_mm_min),
        cusp_markers=bool(include_markers),
        marker_rule="short cusp => larger tactile bump; annotation only",
        implicit=meta,
    )
    return mesh, report


def _write_preview(mesh, path: Path):
    """Simple static PNG preview; matplotlib is optional."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        return False
    # Downsample faces for a responsive preview; STL keeps the full mesh.
    f = mesh.faces
    if len(f) > 35000:
        step = int(math.ceil(len(f)/35000))
        f = f[::step]
    poly = Poly3DCollection(mesh.vertices[f], linewidths=0.0)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(poly)
    lo, hi = mesh.bounds
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try: ax.set_box_aspect(mesh.extents)
    except Exception: pass
    ax.view_init(elev=24, azim=-60)
    ax.set_axis_off()
    ax.set_title("Brooks–Makover tactile model (Euclidean topological realization)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description="Generate a watertight 3-D-printable BM compactification sculpture")
    ap.add_argument("--n", type=int, required=True, help="BM parameter: 2n ideal triangles")
    ap.add_argument("--seed", type=int, default=1, help="same RNG seed as run_bm.py")
    ap.add_argument("--size-mm", type=float, default=160.0, help="target x extent of print in millimetres")
    ap.add_argument("--resolution", type=float, default=5.5,
                    help="marching-cubes voxels per structural tube radius (4=quick, 6-8=final)")
    ap.add_argument("--no-cusp-markers", action="store_true", help="omit tactile cusp-length bumps")
    ap.add_argument("--out", type=str, default=None, help="output prefix (default: bm_print_n<N>_s<seed>)")
    ap.add_argument("--preview", action="store_true", help="also render a PNG preview if matplotlib is installed")
    args = ap.parse_args()

    prefix = Path(args.out or f"bm_print_n{args.n}_s{args.seed}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    mesh, report = printable_model(args.n, args.seed, size_mm=args.size_mm,
                                   resolution=args.resolution,
                                   include_markers=not args.no_cusp_markers)

    stl = prefix.with_suffix(".stl")
    obj = prefix.with_suffix(".obj")
    js = prefix.with_suffix(".json")
    mesh.export(stl)
    mesh.export(obj)
    js.write_text(json.dumps(report, indent=2) + "\n")
    preview = prefix.with_name(prefix.name + "_preview").with_suffix(".png")
    if args.preview:
        _write_preview(mesh, preview)

    print("Brooks–Makover printable model")
    print(f"  sample       n={report['n']} seed={report['seed']}  genus={report['genus_expected']}  cusps={report['cusps']}")
    print(f"  cusp lengths {report['cusp_lengths']}")
    print(f"  mesh         {report['mesh_vertices']} vertices, {report['mesh_faces']} faces")
    print(f"  watertight   {report['watertight']}  winding={report['winding_consistent']}")
    print(f"  Euler/genus  {report['euler_number']} / {report['genus_mesh']} (expected {report['genus_expected']})")
    print("  size [mm]    " + " x ".join(f"{v:.1f}" for v in report['extents_mm']))
    print(f"  tube ~       {report['tube_diameter_mm_estimate']:.2f} mm")
    if report['thin_warning']:
        print("  WARNING      structural tubes are thin for ordinary FDM; increase --size-mm")
    print(f"  STL          {stl}")
    print(f"  OBJ          {obj}")
    print(f"  report       {js}")
    if args.preview and preview.exists(): print(f"  preview      {preview}")

    if not report["watertight"] or report["genus_mesh"] != report["genus_expected"]:
        raise SystemExit("ERROR: topology/mesh validation failed; increase --resolution or adjust geometry")


if __name__ == "__main__":
    main()
