from __future__ import annotations

from pathlib import Path
import time


def ellipsoid_mesh_path(data_dir: str | Path, a: float, b: float, c: float, h: float) -> Path:
    return Path(data_dir) / f"ellipsoid_a{a:g}_b{b:g}_c{c:g}_h{h:g}.msh"


def make_ellipsoid_mesh_gmsh(
    out_msh: str | Path,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    mesh_size: float = 0.08,
    verbose: bool = False,
) -> Path:
    import gmsh

    out_msh = Path(out_msh)
    out_msh.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    try:
        gmsh.model.add("ellipsoid")
        sphere = gmsh.model.occ.addSphere(0, 0, 0, 1.0)
        gmsh.model.occ.dilate([(3, sphere)], 0, 0, 0, a, b, c)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(out_msh))
    finally:
        gmsh.finalize()
    return out_msh


def load_tet_mesh_for_skfem(msh_path: str | Path):
    import meshio
    from skfem import MeshTet

    msh = meshio.read(str(msh_path))
    tetra = None
    for cell_block in msh.cells:
        if cell_block.type in ("tetra", "tetra4"):
            tetra = cell_block.data
            break
    if tetra is None:
        raise ValueError(f"No tetrahedral cells found in mesh: {msh_path}")
    points = msh.points[:, :3]
    return MeshTet(points.T, tetra.T)


def assemble_laplace_dirichlet(mesh, order: int = 2):
    import numpy as np
    from skfem import Basis, ElementTetP1, ElementTetP2, asm, BilinearForm
    from skfem.helpers import dot, grad

    @BilinearForm
    def laplace_form(u, v, w):
        return dot(grad(u), grad(v))

    @BilinearForm
    def mass_form(u, v, w):
        return u * v

    if order == 1:
        element = ElementTetP1()
    elif order == 2:
        element = ElementTetP2()
    else:
        raise ValueError("Only order=1 or order=2 is supported.")

    basis = Basis(mesh, element)
    K = asm(laplace_form, basis)
    M = asm(mass_form, basis)

    boundary_dofs = basis.get_dofs().all()
    all_dofs = np.arange(K.shape[0])
    free = np.setdiff1d(all_dofs, boundary_dofs)

    return K[free][:, free], M[free][:, free]


def load_or_create_problem(
    data_dir: str | Path,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    mesh_size: float = 0.08,
    order: int = 2,
    force_remesh: bool = False,
):
    path = ellipsoid_mesh_path(data_dir, a, b, c, mesh_size)
    if force_remesh or not path.exists():
        make_ellipsoid_mesh_gmsh(path, a=a, b=b, c=c, mesh_size=mesh_size)
    mesh = load_tet_mesh_for_skfem(path)
    t0 = time.perf_counter()
    K, M = assemble_laplace_dirichlet(mesh, order=order)
    return mesh, K, M, {"mesh_path": path, "assembly_sec": time.perf_counter() - t0}
