from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np

from hearing_ellipsoid_bench.solvers import slepc


class _FakeMat:
    def __init__(self):
        self.options = []

    def setOption(self, option, value):
        self.options.append((option, value))


class _FakePC:
    def __init__(self):
        self.pc_type = None

    def setType(self, pc_type):
        self.pc_type = pc_type


class _FakeKSP:
    def __init__(self):
        self.ksp_type = None
        self.pc = _FakePC()

    def setType(self, ksp_type):
        self.ksp_type = ksp_type

    def getPC(self):
        return self.pc


class _FakeST:
    def __init__(self):
        self.st_type = None
        self.ksp = _FakeKSP()

    def setType(self, st_type):
        self.st_type = st_type

    def getKSP(self):
        return self.ksp


class _FakeEPS:
    ErrorType = SimpleNamespace(RELATIVE="relative")
    ProblemType = SimpleNamespace(GHEP="ghep")
    Type = SimpleNamespace(KRYLOVSCHUR="krylovschur")
    Which = SimpleNamespace(ALL="all")
    last = None

    def __init__(self):
        self.calls = {}
        self.st = _FakeST()
        _FakeEPS.last = self

    def create(self):
        return self

    def setOperators(self, K, M):
        self.calls["operators"] = (K, M)

    def setProblemType(self, value):
        self.calls["problem_type"] = value

    def setType(self, value):
        self.calls["type"] = value

    def setInterval(self, lo, hi):
        self.calls["interval"] = (lo, hi)

    def setWhichEigenpairs(self, value):
        self.calls["which"] = value

    def setTolerances(self, **values):
        self.calls["tolerances"] = values

    def setKrylovSchurDimensions(self, **values):
        self.calls["local_dimensions"] = values

    def getST(self):
        return self.st

    def setFromOptions(self):
        self.calls["from_options"] = True

    def solve(self):
        self.calls["solved"] = True

    def getConverged(self):
        return 3

    def getEigenvalue(self, i):
        return [12.0, 4.0, 8.0][i]

    def computeError(self, i, error_type):
        assert error_type == "relative"
        return [1e-12, 2e-12, 3e-12][i]

    def getConvergedReason(self):
        return 1

    def getKrylovSchurInertias(self):
        return [10.0, 0.0, 20.0], [2, 0, 3]


def test_serial_spectrum_slicing_configuration_and_inertia(monkeypatch):
    fake_petsc = SimpleNamespace(
        COMM_WORLD=SimpleNamespace(getSize=lambda: 1),
        Mat=SimpleNamespace(Option=SimpleNamespace(SYMMETRIC="symmetric")),
        KSP=SimpleNamespace(Type=SimpleNamespace(PREONLY="preonly")),
        PC=SimpleNamespace(Type=SimpleNamespace(CHOLESKY="cholesky")),
    )
    fake_slepc = SimpleNamespace(
        EPS=_FakeEPS,
        ST=SimpleNamespace(Type=SimpleNamespace(SINVERT="sinvert")),
    )
    monkeypatch.setitem(sys.modules, "petsc4py", SimpleNamespace(PETSc=fake_petsc))
    monkeypatch.setitem(sys.modules, "slepc4py", SimpleNamespace(SLEPc=fake_slepc))
    matrices = [_FakeMat(), _FakeMat()]
    remaining_matrices = matrices.copy()
    monkeypatch.setattr(
        slepc, "scipy_to_petsc_aij", lambda _A: remaining_matrices.pop(0)
    )

    result = slepc.solve_slepc_spectrum_slicing(
        np.eye(3), np.eye(3), 0.0, 20.0, local_nev=5, local_ncv=10
    )

    assert result.success
    np.testing.assert_array_equal(result.eigs, [4.0, 8.0, 12.0])
    assert result.meta["inertia_count"] == 3
    assert result.meta["count_matches_inertia"] is True
    assert result.meta["slicing_shifts"] == [0.0, 10.0, 20.0]
    assert result.meta["slicing_inertias"] == [0, 2, 3]
    assert result.meta["converged_reason"] == 1
    assert result.meta["residual_rel_max"] == 3e-12
    assert result.meta["residual_rel_median"] == 2e-12
    assert all(mat.options == [("symmetric", True)] for mat in matrices)
    assert _FakeEPS.last.calls["problem_type"] == "ghep"
    assert _FakeEPS.last.calls["type"] == "krylovschur"
    assert _FakeEPS.last.calls["interval"] == (0.0, 20.0)
    assert _FakeEPS.last.calls["which"] == "all"
    assert _FakeEPS.last.calls["local_dimensions"] == {"nev": 5, "ncv": 10}
    assert _FakeEPS.last.st.st_type == "sinvert"
    assert _FakeEPS.last.st.ksp.ksp_type == "preonly"
    assert _FakeEPS.last.st.ksp.pc.pc_type == "cholesky"
    json.dumps(result.meta)


def test_spectrum_slicing_rejects_multiple_mpi_ranks(monkeypatch):
    fake_petsc = SimpleNamespace(COMM_WORLD=SimpleNamespace(getSize=lambda: 2))
    monkeypatch.setitem(sys.modules, "petsc4py", SimpleNamespace(PETSc=fake_petsc))
    monkeypatch.setitem(sys.modules, "slepc4py", SimpleNamespace(SLEPc=object()))

    result = slepc.solve_slepc_spectrum_slicing(np.eye(2), np.eye(2), 0.0, 10.0)

    assert not result.success
    assert result.n_found == 0
    assert "one MPI rank only" in result.message
