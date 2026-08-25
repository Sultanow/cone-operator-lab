from __future__ import annotations

import time
import numpy as np

from hearing_ellipsoid_bench.core.types import AlgorithmResult, clean_eigenvalues


def check_slepc() -> bool:
    try:
        import petsc4py  # noqa: F401
        import slepc4py  # noqa: F401
        return True
    except Exception:
        return False


def scipy_to_petsc_aij(A):
    from petsc4py import PETSc
    from scipy.sparse import csr_matrix

    A = csr_matrix(A)
    mat = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
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

        if PETSc.COMM_WORLD.getSize() != 1:
            raise RuntimeError(
                "Minimal spectrum-slicing implementation supports one MPI rank only."
            )
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
            },
        )
    except Exception as e:
        return AlgorithmResult(
            name,
            clean_eigenvalues([]),
            time.perf_counter() - t0,
            False,
            repr(e),
        )
