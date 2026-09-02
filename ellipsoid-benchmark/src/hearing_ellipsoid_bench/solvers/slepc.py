from __future__ import annotations

import time
import numpy as np
import traceback

from hearing_ellipsoid_bench.core.types import AlgorithmResult, clean_eigenvalues


def check_slepc() -> bool:
    try:
        import petsc4py  # noqa: F401
        import slepc4py  # noqa: F401
        return True
    except Exception:
        return False


def scipy_to_petsc_aij(A, comm=None):
    """Convert a SciPy sparse matrix to a PETSc AIJ matrix.

    Under MPI every rank holds the full SciPy matrix (the finite-element
    assembly is repeated redundantly on each rank) but contributes only
    its own row block to the PETSc matrix, as PETSc requires.
    """
    from petsc4py import PETSc
    from scipy.sparse import csr_matrix
    import numpy as np

    A = csr_matrix(A)
    A.sort_indices()
    n = A.shape[0]
    comm = comm if comm is not None else PETSc.COMM_WORLD

    if comm.getSize() == 1:
        mat = PETSc.Mat().createAIJ(
            size=A.shape,
            csr=(A.indptr.astype(PETSc.IntType),
                 A.indices.astype(PETSc.IntType),
                 A.data),
            comm=comm,
        )
        mat.assemble()
        return mat

    # determine this rank's row ownership from an empty matrix of the
    # same global size, then rebuild with the local CSR block only
    probe = PETSc.Mat().createAIJ(size=(n, n), comm=comm)
    probe.setUp()
    rstart, rend = probe.getOwnershipRange()
    probe.destroy()

    p0, p1 = A.indptr[rstart], A.indptr[rend]
    loc_indptr = (A.indptr[rstart:rend + 1] - p0).astype(PETSc.IntType)
    loc_indices = A.indices[p0:p1].astype(PETSc.IntType)
    loc_data = np.asarray(A.data[p0:p1], dtype=PETSc.ScalarType)

    mat = PETSc.Mat().createAIJ(
        size=((rend - rstart, n), (rend - rstart, n)),
        csr=(loc_indptr, loc_indices, loc_data),
        comm=comm,
    )
    mat.assemble()
    return mat

def solve_slepc_krylov_schur(K, M, n_eigs: int, tol=1e-8, max_it=100_000) -> AlgorithmResult:
    name = "slepc_krylov_schur_lowest"
    t0 = time.perf_counter()
    try:
        from slepc4py import SLEPc

        Kp = scipy_to_petsc_aij(K)
        Mp = scipy_to_petsc_aij(M)

        eps = SLEPc.EPS().create()
        eps.setOperators(Kp, Mp)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eps.setDimensions(n_eigs)
        eps.setTolerances(tol, max_it)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        eps.setFromOptions()
        eps.solve()

        nconv = eps.getConverged()
        vals = [np.real(eps.getEigenvalue(i)) for i in range(min(nconv, n_eigs))]
        return AlgorithmResult(name, clean_eigenvalues(vals), time.perf_counter() - t0, meta={
            "n_eigs_requested": n_eigs,
            "n_converged": int(nconv),
            "tol": tol,
            "max_it": max_it,
        })
    except Exception as e:
        return AlgorithmResult(name, clean_eigenvalues([]), time.perf_counter() - t0, False, repr(e))


def solve_slepc_spectrum_slicing(
    K,
    M,
    lambda_min: float,
    lambda_max: float,
    tol: float = 1e-10,
    max_it: int = 100_000,
    local_nev: int = 80,
    local_ncv: int = 160,
    partitions: int = 1,
    raise_on_error: bool = False,
) -> AlgorithmResult:
    """Compute all generalized Hermitian eigenvalues in an interval.

    This is the deliberately serial reference implementation used for the
    comparison with the block-Hankel method.  Krylov-Schur spectrum slicing
    uses shift-and-invert and inertia to certify the number of eigenvalues in
    ``[lambda_min, lambda_max]``.
    """
    name = "slepc_krylovschur_spectrum_slicing"
    t0 = time.perf_counter()

    try:
        from petsc4py import PETSc
        from slepc4py import SLEPc

        try:
            PETSc.Sys.pushErrorHandler("traceback")
        except Exception:
            pass
        
        size = PETSc.COMM_WORLD.getSize()
        if partitions < 1 or partitions > size:
            raise ValueError(
                f"partitions must satisfy 1 <= partitions <= {size}")
        if size % partitions != 0:
            raise ValueError(
                f"MPI size {size} is not divisible by partitions {partitions}")
        if not lambda_min < lambda_max:
            raise ValueError("lambda_min must be smaller than lambda_max")
        if local_nev < 1 or local_ncv <= local_nev:
            raise ValueError("require 1 <= local_nev < local_ncv")

        Kp = scipy_to_petsc_aij(K)
        Mp = scipy_to_petsc_aij(M)
        Kp.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        Mp.setOption(PETSc.Mat.Option.SYMMETRIC, True)

        eps = SLEPc.EPS().create()
        eps.setOperators(Kp, Mp)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eps.setInterval(lambda_min, lambda_max)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.ALL)
        eps.setTolerances(tol=tol, max_it=max_it)

        # Unlike EPS.setDimensions(), these dimensions apply to each local
        # slicing subsolve.  They are large enough for the unit-ball
        # multiplicities in the interval used in the paper.
        eps.setKrylovSchurDimensions(nev=local_nev, ncv=local_ncv)
        if partitions > 1:
            eps.setKrylovSchurPartitions(partitions)

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        ksp = st.getKSP()
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.CHOLESKY)

        # In particular, this permits selecting MUMPS from the command line.
        eps.setFromOptions()
        eps.solve()

        nconv = eps.getConverged()
        vals = clean_eigenvalues(
            [np.real(eps.getEigenvalue(i)) for i in range(nconv)]
        )
        relative_errors = np.asarray(
            [
                eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
                for i in range(nconv)
            ],
            dtype=float,
        )

        shifts, inertias = eps.getKrylovSchurInertias()
        shifts = np.asarray(shifts, dtype=float)
        inertias = np.asarray(inertias, dtype=int)
        order = np.argsort(shifts)
        shifts = shifts[order]
        inertias = inertias[order]
        inertia_count = (
            int(inertias[-1] - inertias[0]) if len(inertias) >= 2 else None
        )

        return AlgorithmResult(
            name,
            vals,
            time.perf_counter() - t0,
            True,
            "",
            meta={
                "lambda_min": float(lambda_min),
                "lambda_max": float(lambda_max),
                "n_converged": int(nconv),
                "converged_reason": int(eps.getConvergedReason()),
                "inertia_count": inertia_count,
                "count_matches_inertia": (
                    inertia_count == int(nconv) if inertia_count is not None else None
                ),
                "tol": tol,
                "max_it": max_it,
                "local_nev": local_nev,
                "local_ncv": local_ncv,
                "residual_rel_max": (
                    float(relative_errors.max()) if len(relative_errors) else None
                ),
                "residual_rel_median": (
                    float(np.median(relative_errors)) if len(relative_errors) else None
                ),
                "slicing_shifts": shifts.tolist(),
                "slicing_inertias": inertias.tolist(),
                "mpi_size": int(size),
                "partitions": int(partitions),
            },
        )
    except Exception as e:
        detail = traceback.format_exc()

        # PETSc-Fehler tragen einen numerischen Code, der die Ursache
        # benennt (z.B. 76 = Fehler in externer Bibliothek wie MUMPS,
        # 55 = zu wenig Speicher). Cython verpackt das gelegentlich in
        # einen generischen SystemError, dann steht die eigentliche
        # Ursache in der Exception-Kette.
        parts = [repr(e)]
        ierr = getattr(e, "ierr", None)
        if ierr is not None:
            parts.append(f"PETSc error code {ierr}")
        cause = e.__cause__ or e.__context__
        while cause is not None:
            parts.append(f"caused by: {cause!r}")
            c_ierr = getattr(cause, "ierr", None)
            if c_ierr is not None:
                parts.append(f"PETSc error code {c_ierr}")
            cause = cause.__cause__ or cause.__context__

        message = " | ".join(parts) + "\n" + detail

        if raise_on_error:
            raise RuntimeError(message) from e

        return AlgorithmResult(
            name,
            clean_eigenvalues([]),
            time.perf_counter() - t0,
            False,
            message,
            meta={
                "lambda_min": float(lambda_min),
                "lambda_max": float(lambda_max),
                "local_nev": local_nev,
                "local_ncv": local_ncv,
                "traceback": detail,
            },
        )

