from mpi4py import MPI
import numpy as np

import gmsh
from dolfinx import fem, io
from dolfinx.io import gmshio
from dolfinx.fem.petsc import assemble_matrix
import ufl
from petsc4py import PETSc
from slepc4py import SLEPc


def generate_ellipsoid_mesh(a=1.0, b=1.5, c=2.3, lc=0.2, comm=MPI.COMM_WORLD):
    """
    Erzeugt ein Ellipsoid-Mesh via Gmsh und konvertiert es direkt
    nach dolfinx.mesh.Mesh mit gmshio.model_to_mesh.
    """
    rank = comm.rank

    if rank == 0:
        print(f"Erzeuge Ellipsoid-Mesh mit a={a}, b={b}, c={c}, lc={lc}")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("ellipsoid")

    # 1. Einheitskugel
    sph = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
    gmsh.model.occ.synchronize()

    # 2. Skalierung: Kugel -> Ellipsoid
    gmsh.model.occ.dilate([(3, sph)], 0.0, 0.0, 0.0, a, b, c)
    gmsh.model.occ.synchronize()

    # 3. Physikalische Volumengruppe definieren
    vols = gmsh.model.getEntities(dim=3)
    vol_tags = [v[1] for v in vols]
    gmsh.model.addPhysicalGroup(3, vol_tags, tag=1)
    gmsh.model.setPhysicalName(3, 1, "EllipsoidVolume")

    # 4. Oberflächen-Physikalische Gruppe (für spätere Randbedingungen)
    surfs = gmsh.model.getEntities(dim=2)
    surf_tags = [s[1] for s in surfs]
    gmsh.model.addPhysicalGroup(2, surf_tags, tag=2)
    gmsh.model.setPhysicalName(2, 2, "EllipsoidSurface")

    # 5. Meshing
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(2) 

    # 6. Konvertierung zu dolfinx-Mesh
    domain, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=3
    )

    if rank == 0:
        print("Mesh-Knoten:", domain.geometry.x.shape[0])

    gmsh.finalize()
    return domain, cell_tags, facet_tags


def compute_ellipsoid_eigenvalues(
    a=1.0,
    b=1.5,
    c=2.3,
    lc=0.1,
    num_eigs=20,
    chunk_size=10,
    out_prefix="ellipsoid_eigs",
):

    comm = MPI.COMM_WORLD
    rank = comm.rank

    # 1. Mesh erzeugen
    domain, cell_tags, facet_tags = generate_ellipsoid_mesh(
        a=a, b=b, c=c, lc=lc, comm=comm
    )

    # 2. Funktionsraum
    V = fem.functionspace(domain, ("Lagrange", 2))

    # 3. Dirichlet-Rand: u = 0 auf der Ellipsoidoberfläche
    #    => wir nehmen alle Rand-Dofs (Facetten auf dem äußeren Rand)
    facet_dim = domain.topology.dim - 1
    domain.topology.create_connectivity(facet_dim, 0)

    def boundary_indicator(x):
        # Ellipsoid-Gleichung (numerisch): (x/a)^2 + (y/b)^2 + (z/c)^2 ≈ 1
        return np.isclose(
            (x[0] / a) ** 2 + (x[1] / b) ** 2 + (x[2] / c) ** 2,
            1.0,
            atol=1e-6,
        )

    bdofs = fem.locate_dofs_geometrical(V, boundary_indicator)
    bc = fem.dirichletbc(PETSc.ScalarType(0), bdofs, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a_form = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    m_form = fem.form(u * v * ufl.dx)

    A = assemble_matrix(a_form)
    A.assemble()
    M = assemble_matrix(m_form)
    M.assemble()

    # Rand-DOFs (Dirichlet) aus locate_dofs_geometrical
    bdofs = fem.locate_dofs_geometrical(V, boundary_indicator)
    bdofs = np.unique(bdofs)

    # Alle DOFs (lokal, bei 1 Prozess = global)
    n_dofs = A.getSize()[0]
    all_dofs = np.arange(n_dofs, dtype=np.int32)

    # Innere DOFs = alle außer Rand
    inner_dofs = np.setdiff1d(all_dofs, bdofs)

    if rank == 0:
        print("Matrixgröße A:", A.getSize())
        print("Anzahl Rand-DOFs:", len(bdofs))
        print("Anzahl innere DOFs:", len(inner_dofs))

    # Submatrix nur auf inneren DOFs
    is_inner = PETSc.IS().createGeneral(inner_dofs, comm=comm)

    A_sub = A.createSubMatrix(is_inner, is_inner)
    M_sub = M.createSubMatrix(is_inner, is_inner)

    # 4. Eigenwertproblem A_sub x = λ M_sub x
    eps = SLEPc.EPS().create(comm)
    eps.setOperators(A_sub, M_sub)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)

    # Robuster iterativer Solver
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

    # Kein Shift-and-Invert, nur einfacher Shift (σ = 0)
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SHIFT)
    st.setShift(0.0)

    # Iterativer linearer Solver statt LU/MUMPS
    opts = PETSc.Options()
    opts["st_ksp_type"] = "cg"
    opts["st_pc_type"] = "gamg"

    eps.setDimensions(num_eigs, PETSc.DECIDE)
    eps.setTolerances(1e-9, 10000)

    if rank == 0:
        print("Löse Eigenwertproblem für", num_eigs, "Eigenwerte ...")

    eps.solve()


    nconv = eps.getConverged()
    k = min(nconv, num_eigs)

    if rank == 0:
        print("Konvergiert:", nconv, "Eigenwerte, verwende:", k)

    lambdas = []
    for i in range(k):
        lam = eps.getEigenvalue(i)
        lambdas.append(lam)

    lambdas = np.sort(np.real(np.array(lambdas)))

    if rank == 0:
        print("Erste 10 Eigenwerte:", lambdas[:10])
        if k > 10:
            print("Letzte 10 Eigenwerte:", lambdas[-10:])

        # Komplett speichern
        all_file = f"{out_prefix}_a{a}_b{b}_c{c}_N{k}.txt"
        np.savetxt(all_file, lambdas, fmt="%.15e")
        print("Alle Eigenwerte gespeichert in:", all_file)

        # Chunkweise speichern
        if chunk_size and chunk_size > 0:
            num_chunks = int(np.ceil(k / chunk_size))
            for j in range(num_chunks):
                start = j * chunk_size
                end = min((j + 1) * chunk_size, k)
                chunk = lambdas[start:end]
                chunk_file = (
                    f"{out_prefix}_a{a}_b{b}_c{c}_"
                    f"k{start+1:04d}_k{end:04d}.txt"
                )
                np.savetxt(chunk_file, chunk, fmt="%.15e")
            print(f"Chunks geschrieben (je {chunk_size} EW): {num_chunks} Dateien")

    return lambdas


if __name__ == "__main__":
    a_true = 1.0
    b_true = 1.5
    c_true = 2.3

    # Meshauflösung: lc kleiner => feineres Mesh
    lc = 0.08

    N_EIG = 20
    CHUNK = 10

    compute_ellipsoid_eigenvalues(
        a=a_true,
        b=b_true,
        c=c_true,
        lc=lc,
        num_eigs=N_EIG,
        chunk_size=CHUNK,
        out_prefix="ellipsoid",
    )
