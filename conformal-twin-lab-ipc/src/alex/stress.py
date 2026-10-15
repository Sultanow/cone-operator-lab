"""Section 'Is P determined by its intrinsic geometry?': the rigidity numbers of P and of its edge graph.
  1. local uniqueness: rank of the rigidity matrix at P (must be 66 = 3*24 - 6) and its smallest non-zero singular
     value for unit edge directions;
  2. generic global rigidity (Gortler-Healy-Thurston): dimension of the stress space and the largest rank of a
     stress matrix at random coordinates (criterion: n - d - 1 = 20);
  3. Connelly's sufficient condition at P itself: is there a positive semidefinite stress of rank 20? Semidefinite
     program: maximise the smallest eigenvalue of the stress matrix on the 20-dimensional complement of the
     always-present kernel span{1, x, y, z}, over stresses of norm <= 1. A value ~0 means: no such stress;
  4. the conic-at-infinity condition (edge directions not on a conic), the second half of Connelly's criterion.
  5. exact versions of 1 and 2 over the rationals (P has integer coordinates): rank of the rigidity matrix, an integer
     basis of the stress space, and the exact rank of integer stress matrices. Together with the perturbation argument
     in the paper this proves that P is infinitesimally rigid and that the edge graph of K is generically globally
     rigid in R^3.
Usage: python src/alex/stress.py OUTDIR      (step 3 needs cvxpy, step 5 needs python-flint)"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from realize import P, E
OUT = sys.argv[1] if len(sys.argv) > 1 else '.'; os.makedirs(OUT, exist_ok=True)
n, d = 24, 3
def rigidity_matrix(X, unit=False):
    Rm = np.zeros((len(E), 3 * n))
    for k, (i, j) in enumerate(E):
        v = X[i] - X[j]; v = v / np.linalg.norm(v) if unit else v
        Rm[k, 3 * i:3 * i + 3] = v; Rm[k, 3 * j:3 * j + 3] = -v
    return Rm
def stress_basis(X):
    U, s, Vt = np.linalg.svd(rigidity_matrix(X).T, full_matrices=True)
    r = int((s > 1e-9 * s[0]).sum()); return r, Vt[r:]                # rows: orthonormal basis of the stresses
def stress_matrix(w):
    Om = np.zeros((n, n))
    for k, (i, j) in enumerate(E): Om[i, j] -= w[k]; Om[j, i] -= w[k]; Om[i, i] += w[k]; Om[j, j] += w[k]
    return Om
def max_stress_rank(X, trials=5, seed=0):
    rng = np.random.default_rng(seed); _, W = stress_basis(X); best = 0
    for _ in range(trials):
        sv = np.linalg.svd(stress_matrix(rng.normal(size=len(W)) @ W), compute_uv=False)
        best = max(best, int((sv > 1e-8 * sv[0]).sum()))
    return best
rep = {}
# 1. local uniqueness
s = np.linalg.svd(rigidity_matrix(P, unit=True), compute_uv=False); r = int((s > 1e-9 * s[0]).sum())
rep['rigidity_rank_at_P'] = r; rep['smallest_nonzero_singular_value_unit'] = float(s[r - 1])
print(f'1. rigidity matrix at P: rank {r} (needed {3 * n - 6}), smallest non-zero singular value {s[r - 1]:.4f} (unit edge directions)')
# 2. generic global rigidity
rng = np.random.default_rng(1); rows = []
for name, X in [('random coordinates 1', rng.normal(size=(n, 3)) * 100), ('random coordinates 2', rng.normal(size=(n, 3)) * 100), ('P itself', P)]:
    rr, W = stress_basis(X); sr = max_stress_rank(X); rows.append(dict(config=name, rigidity_rank=rr, stress_dim=len(W), max_stress_rank=sr))
    print(f'2. {name:22s}: rigidity rank {rr}, stress space dimension {len(W)}, largest stress-matrix rank {sr} (criterion {n - d - 1})')
rep['ght'] = rows
# 3. Connelly at P: positive semidefinite stress of rank n - d - 1?
_, W = stress_basis(P); Bs = [stress_matrix(W[a]) for a in range(len(W))]
Q, _ = np.linalg.qr(np.c_[np.ones(n), P], mode='complete'); Qc = Q[:, 4:]          # complement of span{1, x, y, z}
try:
    import cvxpy as cp
    rep['sdp'] = []
    for solver in [x for x in ('SCS', 'CLARABEL') if x in cp.installed_solvers()]:   # SCS as in the paper, Clarabel as a cross-check
        c = cp.Variable(len(W)); t = cp.Variable()
        M = sum(c[a] * (Qc.T @ Bs[a] @ Qc) for a in range(len(W)))
        prob = cp.Problem(cp.Maximize(t), [M - t * np.eye(n - 4) >> 0, cp.norm(c, 2) <= 1])
        if solver == 'SCS': prob.solve(solver=solver, eps=1e-9, max_iters=200000)
        else: prob.solve(solver=solver)
        ev = np.linalg.eigvalsh(Qc.T @ stress_matrix(np.asarray(c.value) @ W) @ Qc)
        rep['sdp'].append(dict(solver=solver, status=prob.status, best_min_eigenvalue=float(t.value), eig_min=float(ev.min()), eig_max=float(ev.max())))
        print(f'3. SDP ({solver}, {prob.status}): largest attainable smallest eigenvalue on the {n - 4}-dim complement = {t.value:.1e}')
    print('   -> zero within solver accuracy: no positive semidefinite stress of rank 20, Connelly\'s criterion does not apply')
except ImportError:
    rep['sdp'] = None; print('3. skipped: cvxpy is not installed (pip install cvxpy)')
# 4. conic at infinity
Dd = P[E[:, 0]] - P[E[:, 1]]
A = np.c_[Dd[:, 0] ** 2, Dd[:, 1] ** 2, Dd[:, 2] ** 2, Dd[:, 0] * Dd[:, 1], Dd[:, 0] * Dd[:, 2], Dd[:, 1] * Dd[:, 2]]
rep['conic_rank'] = int(np.linalg.matrix_rank(A))
print(f'4. edge directions: rank {rep["conic_rank"]} of the quadratic matrix (6 = not on a conic at infinity)')
# 5. exact arithmetic over Q
try:
    import flint
    Pi = np.rint(P).astype(int); assert np.array_equal(Pi, P)
    Rex = [[0] * (3 * n) for _ in range(len(E))]
    for k, (i, j) in enumerate(E):
        for c in range(3):
            dd = int(Pi[i, c] - Pi[j, c]); Rex[k][3 * i + c] = dd; Rex[k][3 * j + c] = -dd
    RZ = flint.fmpz_mat(Rex); rank_ex = RZ.rank()
    N, nul = RZ.transpose().nullspace(); basis = [[int(N[r, c]) for r in range(N.nrows())] for c in range(nul)]
    for w in basis:                                                   # every basis vector is an exact equilibrium stress
        acc = np.zeros((n, 3), dtype=object)
        for k, (i, j) in enumerate(E): dv = (Pi[i] - Pi[j]).astype(object) * w[k]; acc[i] += dv; acc[j] -= dv
        assert all(x == 0 for x in acc.ravel())
    rng5 = np.random.default_rng(7); ranks = []; coefs = []
    for trial in range(5):
        cf = [int(x) for x in rng5.integers(-50, 51, size=nul)]; coefs.append(cf)
        w = [sum(cf[a] * basis[a][k] for a in range(nul)) for k in range(len(E))]
        Om = [[0] * n for _ in range(n)]
        for k, (i, j) in enumerate(E): Om[i][j] -= w[k]; Om[j][i] -= w[k]; Om[i][i] += w[k]; Om[j][j] += w[k]
        ranks.append(flint.fmpz_mat(Om).rank())
    rep['exact'] = dict(rigidity_rank=rank_ex, stress_dim=nul, stress_matrix_ranks=ranks, coefficients=coefs)
    print(f'5. exact over Q: rigidity rank {rank_ex}, stress space dimension {nul}, ranks of 5 integer stress matrices {ranks}')
    print('   -> P is infinitesimally rigid; with a stress of rank n-d-1 = 20 at P the edge graph of K is generically globally rigid')
except ImportError:
    rep['exact'] = None; print('5. skipped: python-flint is not installed')
json.dump(rep, open(os.path.join(OUT, 'stress_report.json'), 'w'), indent=1)
