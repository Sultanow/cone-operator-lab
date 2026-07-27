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
