# scripts/run_slepc_slicing.py
"""Inertia-based spectrum slicing with SLEPc/MUMPS on the SAME (K, M)
matrices used by the other solvers in this benchmark.

SLEPc's interval mode (``EPS_ALL`` with ``setInterval``) partitions a
prescribed interval into slices, runs a shifted Lanczos process per
slice, and uses the inertia of ``K - sigma M`` obtained from the
MUMPS ``LDL^T`` factorization to guarantee that no eigenvalue inside
the interval is missed and that multiplicities are correct. This is
the established reference for "complete spectrum in an interval" and
therefore the natural comparison for the gap-free block-Hankel method.

The script deliberately reuses ``load_or_create_problem`` so that the
mesh, element order, and assembled matrices are bitwise identical to
the FEM--ARPACK, block-Hankel, and certification runs.

Requirements (module or conda environment):
    petsc, slepc4py, petsc4py, mumps

Example (single task, whole interval):
    python scripts/run_slepc_slicing.py \
        --data-root /home/esul01/data \
        --out-dir   /home/esul01/data/outputs/slepc_slicing_triaxial \
        --a 1.0 --b 1.5 --c 2.3 --mesh-h 0.06 --order 2 \
        --lambda-lo 0.0 --lambda-hi 449.1 --partitions 1

Example (MPI with spectrum partitioning over 8 sub-communicators):
    mpirun -n 64 python scripts/run_slepc_slicing.py ... --partitions 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import slepc4py
import petsc4py


def to_petsc_aij(A_csr, comm):
    """Wrap a SciPy CSR matrix as a PETSc AIJ matrix (rank 0 holds it)."""
    from petsc4py import PETSc

    A_csr = A_csr.tocsr()
    A_csr.sort_indices()
    n = A_csr.shape[0]
    P = PETSc.Mat().createAIJ(
        size=(n, n),
        csr=(A_csr.indptr.astype(PETSc.IntType),
             A_csr.indices.astype(PETSc.IntType),
             A_csr.data.astype(PETSc.ScalarType)),
        comm=comm,
    )
    P.assemble()
    P.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    return P


def main():
    petsc4py.init()
    slepc4py.init()
    from petsc4py import PETSc
    from slepc4py import SLEPc

    comm = PETSc.COMM_WORLD
    rank = comm.getRank()

    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)

    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--b", type=float, default=1.5)
    p.add_argument("--c", type=float, default=2.3)
    p.add_argument("--mesh-h", type=float, default=0.06)
    p.add_argument("--order", type=int, default=2)

    p.add_argument("--lambda-lo", type=float, default=0.0,
                   help="lower end of the interval; must be BELOW the "
                        "smallest eigenvalue or inside a spectral gap")
    p.add_argument("--lambda-hi", type=float, required=True)
    p.add_argument("--partitions", type=int, default=1,
                   help="number of MPI sub-communicators SLEPc uses to "
                        "process slices concurrently (spectrum "
                        "partitioning); requires MPI size divisible by it")
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--mpd", type=int, default=None,
                   help="maximum projected dimension per slice; SLEPc "
                        "picks a default when omitted")
    args = p.parse_args()

    run_label = (f"a{args.a:g}_b{args.b:g}_c{args.c:g}"
                 f"_P{args.order}_h{args.mesh_h:g}_slepc_slicing")
    if rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        print("Run label:", run_label, flush=True)
        print("Arguments:", vars(args), flush=True)

    # ---- identical discretization as every other FEM-based run -------
    from hearing_ellipsoid_bench.fem.assembly import load_or_create_problem

    t_assemble = time.perf_counter()
    mesh, K, M, fem_meta = load_or_create_problem(
        args.data_root, a=args.a, b=args.b, c=args.c,
        mesh_size=args.mesh_h, order=args.order, force_remesh=False,
    )
    t_assemble = time.perf_counter() - t_assemble
    n_dofs = K.shape[0]
    if rank == 0:
        print(f"dofs: {n_dofs}  (assembly/load {t_assemble:.1f}s)",
              flush=True)

    A = to_petsc_aij(sp.csr_matrix(K), comm)
    B = to_petsc_aij(sp.csr_matrix(M), comm)

    # ---- SLEPc: all eigenvalues in an interval, inertia-based --------
    eps = SLEPc.EPS().create(comm=comm)
    eps.setOperators(A, B)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.ALL)
    eps.setInterval(args.lambda_lo, args.lambda_hi)
    eps.setTolerances(tol=args.tol)
    if args.mpd:
        eps.setDimensions(mpd=args.mpd)

    # Krylov-Schur with spectrum slicing; inertia comes from the
    # LDL^T factorization performed by MUMPS through the ST object.
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setKrylovSchurPartitions(max(1, args.partitions))

    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    ksp = st.getKSP()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("cholesky")
    pc.setFactorSolverType("mumps")
    # MUMPS must be allowed to handle a factorization with negative
    # pivots; ICNTL(13)=1 turns off ScaLAPACK on the root node, which
    # SLEPc requires in order to read the inertia reliably.
    PETSc.Options().setValue("-mat_mumps_icntl_13", "1")
    PETSc.Options().setValue("-mat_mumps_icntl_24", "1")
    PETSc.Options().setValue("-mat_mumps_cntl_3", "1e-12")
    eps.setFromOptions()

    t0 = time.perf_counter()
    eps.solve()
    wall = time.perf_counter() - t0

    nconv = eps.getConverged()
    if rank == 0:
        print(f"converged: {nconv} eigenvalues in "
              f"[{args.lambda_lo}, {args.lambda_hi}]  "
              f"({wall:.1f}s)", flush=True)

    vals = np.empty(nconv, dtype=float)
    xr = A.createVecRight()
    xi = A.createVecRight()
    res = np.empty(nconv, dtype=float)
    for i in range(nconv):
        vals[i] = eps.getEigenpair(i, xr, xi).real
        res[i] = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
    order = np.argsort(vals)
    vals, res = vals[order], res[order]

    if rank == 0:
        eig_path = args.out_dir / f"{run_label}_eigs.txt"
        np.savetxt(eig_path, vals, fmt="%.17e")

        try:
            inertia_lo, inertia_hi = eps.getKrylovSchurInertias()[:2]
        except Exception:
            inertia_lo = inertia_hi = None

        meta = {
            "solver": "slepc EPS krylovschur, spectrum slicing",
            "which": "EPS_ALL",
            "interval": [args.lambda_lo, args.lambda_hi],
            "partitions": args.partitions,
            "tol": args.tol,
            "mpd": args.mpd,
            "n_dofs": int(n_dofs),
            "n_converged": int(nconv),
            "wall_sec": wall,
            "assemble_sec": t_assemble,
            "mpi_size": int(comm.getSize()),
            "lambda_min": float(vals[0]) if nconv else None,
            "lambda_max": float(vals[-1]) if nconv else None,
            "residual_rel_max": float(res.max()) if nconv else None,
            "residual_rel_median": float(np.median(res)) if nconv else None,
            "inertia_endpoints": [inertia_lo, inertia_hi],
            "fem_meta": {k: v for k, v in (fem_meta or {}).items()
                         if isinstance(v, (int, float, str, bool))},
        }
        (args.out_dir / f"{run_label}_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2))
        print(f"wrote {eig_path}")

    eps.destroy()


if __name__ == "__main__":
    main()
