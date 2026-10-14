"""Search for a C4-equivariant embedding f: (S, h_cp) -> R^3 that is discretely conformal
to the circle-packing hyperbolic twin, smooth (small Willmore energy), free of
self-intersections and close to Mizhaev's polyhedron P.  Multilevel: solve on level
l, prolongate to l+1, continue.

Discrete conformality (Bobenko-Pinkall-Springborn 2015): a Euclidean mesh with edge
lengths l_ij is discretely conformal to the hyperbolic mesh with lengths L_ij iff
    log l_ij = (u_i + u_j)/2 + log(2 sinh(L_ij/2))   for some u: V -> R.
Energy: w_c E_conf + w_fit E_fit + w_will W + w_rep E_barrier, penalty continuation in w_c.
"""
import argparse, json, time, os, sys
import numpy as np, jax, jax.numpy as jnp
import scipy.sparse as sp, scipy.sparse.linalg as spl
from scipy.optimize import minimize
from scipy.spatial import cKDTree
jax.config.update('jax_enable_x64', True)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from fine_mesh import build, prolongation
from geom_np import project_to_P, self_intersections
from ipc import Contact

ap = argparse.ArgumentParser()
ap.add_argument('--levels', type=str, default='2')
ap.add_argument('--w-fit', type=float, default=1.0)
ap.add_argument('--w-will', type=float, default=1e-3)
ap.add_argument('--w-rep', type=float, default=10.0)
ap.add_argument('--barrier', choices=['on', 'off'], default='on')
ap.add_argument('--target', choices=['cp', 'P'], default='cp', help='cp = circle-packing twin h_cp; P = the conformal structure of the polyhedron itself (h_P)')
ap.add_argument('--solver', choices=['lbfgs', 'ipc'], default='lbfgs', help='ipc = contact-aware L-BFGS with CCD line search')
ap.add_argument('--ipc-dhat', type=float, default=0.25, help='IPC barrier range in median fine edge lengths')
ap.add_argument('--dhat', type=float, default=2.0, help='barrier range in median fine edge lengths')
ap.add_argument('--wc-schedule', type=str, default='1,10,100,1e3,1e4,1e5,1e6')
ap.add_argument('--iters', type=int, default=400, help='L-BFGS iterations per continuation stage')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--noise', type=float, default=0.0, help='initial perturbation in fine edge lengths')
ap.add_argument('--out', type=str, default='out')
ap.add_argument('--resume', action='store_true', help='skip levels whose checkpoint already exists')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
tag = f'wf{args.w_fit:g}_ww{args.w_will:g}_wr{args.w_rep:g}_b{args.barrier}_s{args.seed}_n{args.noise:g}' + ('_ipc' if args.solver == 'ipc' else '') + ('_tP' if args.target == 'P' else '')
logf = open(f'{args.out}/{tag}.log', 'a')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()

R = json.load(open(f'{HERE}/results.json')); D = json.load(open(f'{HERE}/development.json'))
VP = np.array(R['vertices'], float) / 300.0
F0 = np.array([D['orient'][str(n)] for n in range(56)])
T = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1.]])

def _snorm(v):
    # norm with a finite gradient at 0 (padding pairs are degenerate; 0 * NaN would poison the gradient)
    return jnp.sqrt(jnp.sum(v * v, -1) + 1e-30)

def pt_tri_dist(p, a, b, c):
    ab, ac = b - a, c - a; n = jnp.cross(ab, ac); n = n / _snorm(n)[..., None]; ap_ = p - a
    d00 = jnp.sum(ab * ab, -1); d01 = jnp.sum(ab * ac, -1); d11 = jnp.sum(ac * ac, -1)
    d20 = jnp.sum(ap_ * ab, -1); d21 = jnp.sum(ap_ * ac, -1); den = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / den; w = (d00 * d21 - d01 * d20) / den
    inside = (v >= 0) & (w >= 0) & (v + w <= 1); dplane = jnp.abs(jnp.sum(ap_ * n, -1))
    def seg(p, x, y):
        t = jnp.clip(jnp.sum((p - x) * (y - x), -1) / jnp.sum((y - x) ** 2, -1), 0.0, 1.0)
        return _snorm(p - (x + t[..., None] * (y - x)))
    return jnp.where(inside, dplane, jnp.minimum(jnp.minimum(seg(p, a, b), seg(p, b, c)), seg(p, c, a)))

def ee_dist_j(P1, Q1, P2, Q2):
    """segment-segment distance as min(interior critical point, four endpoint-to-segment distances):
    exact, and piecewise smooth also for (nearly) parallel edges"""
    d1, d2, r = Q1 - P1, Q2 - P2, P1 - P2
    a = jnp.sum(d1 * d1, -1); e = jnp.sum(d2 * d2, -1); f = jnp.sum(d2 * r, -1); c = jnp.sum(d1 * r, -1); b = jnp.sum(d1 * d2, -1)
    den = a * e - b * b; ok = den > 1e-10 * a * e; dd = jnp.where(ok, den, 1.0)
    s = (b * f - c * e) / dd; t = (a * f - b * c) / dd
    inner = ok & (s > 0) & (s < 1) & (t > 0) & (t < 1)
    s_ = jnp.where(inner, s, 0.5); t_ = jnp.where(inner, t, 0.5)
    dint = jnp.where(inner, _snorm(P1 + d1 * s_[..., None] - (P2 + d2 * t_[..., None])), jnp.inf)
    def ps(p, x, y):
        tt = jnp.clip(jnp.sum((p - x) * (y - x), -1) / jnp.sum((y - x) ** 2, -1), 0.0, 1.0)
        return _snorm(p - (x + tt[..., None] * (y - x)))
    return jnp.minimum(jnp.minimum(jnp.minimum(dint, ps(P1, P2, Q2)), jnp.minimum(ps(Q1, P2, Q2), ps(P2, P1, Q1))), ps(Q2, P1, Q1))

class Level:
    def __init__(self, level):
        M = build(level, HERE); self.M = M
        self.F, self.E, self.tau = M['F'], M['edges'], M['tau']; self.XP = M['XP'] / 300.0; nV = self.nV = len(self.XP)
        if args.target == 'P':
            # discrete conformal equivalence is an equivalence relation on a fixed mesh: being discretely conformal
            # to h_P (P's uniformised metric) is the same as being discretely conformal to P's own edge lengths
            self.logLhat = np.log(np.linalg.norm(self.XP[self.E[:, 0]] - self.XP[self.E[:, 1]], axis=1))
        else:
            self.logLhat = np.log(2 * np.sinh(M['lenH'] / 2))
        tauk = [np.arange(nV)]
        for _ in range(3): tauk.append(self.tau[tauk[-1]])
        nbr = [set() for _ in range(nV)]
        for a, b in self.E: nbr[a].add(b); nbr[b].add(a)
        self.near = []
        for v in range(nV):
            s = set(nbr[v]); fr = set(nbr[v])
            for _ in range(2): fr = set().union(*(nbr[w] for w in fr)) - s - {v}; s |= fr
            self.near.append(s)
        self.h0 = np.median(np.linalg.norm(self.XP[self.E[:, 0]] - self.XP[self.E[:, 1]], axis=1)); dhat = self.dhat = args.dhat * self.h0
        Fi, Fj, Fk = (jnp.array(self.F[:, c]) for c in range(3)); Ei, Ej = jnp.array(self.E[:, 0]), jnp.array(self.E[:, 1])
        jL = jnp.array(self.logLhat); jtau = [jnp.array(t) for t in tauk]
        jTinv = [jnp.array(np.linalg.matrix_power(T.T, k)) for k in range(4)]
        sym = lambda Y: sum(Y[jtau[k]] @ jTinv[k].T for k in range(4)) / 4.0
        symu = lambda w: sum(w[jtau[k]] for k in range(4)) / 4.0
        self.unpack = unpack = lambda z: (sym(z[:3 * nV].reshape(nV, 3)), symu(z[3 * nV:]))
        def willmore(X):
            a, b, c = X[Fi], X[Fj], X[Fk]
            def cot(p, q, r):
                u_, v_ = q - p, r - p; return jnp.sum(u_ * v_, -1) / jnp.linalg.norm(jnp.cross(u_, v_), axis=-1)
            A = 0.5 * jnp.linalg.norm(jnp.cross(b - a, c - a), axis=-1)
            LX = jnp.zeros_like(X)
            for (i, j, cw) in ((Fj, Fk, cot(a, b, c)), (Fk, Fi, cot(b, c, a)), (Fi, Fj, cot(c, a, b))):
                d = (X[j] - X[i]) * (0.5 * cw)[:, None]; LX = LX.at[i].add(d).at[j].add(-d)
            Av = jnp.zeros(nV).at[Fi].add(A / 3).at[Fj].add(A / 3).at[Fk].add(A / 3)
            return jnp.sum(jnp.sum(LX ** 2, -1) / (4 * Av)), Av
        self.willmore = willmore
        self.conf_res = conf_res = lambda X, u: jnp.log(jnp.linalg.norm(X[Ei] - X[Ej], axis=-1)) - 0.5 * (u[Ei] + u[Ej]) - jL
        use_bar = args.barrier == 'on'
        def energy(z, wc, tgt, nrm, pi, pj):
            X, u = unpack(z); r = conf_res(X, u); Ec = jnp.mean(r ** 2); W, Av = willmore(X); dv = X - tgt
            Ef = jnp.sum(Av * (jnp.sum(dv * nrm, -1) ** 2 + 0.05 * jnp.sum(dv ** 2, -1))) / jnp.sum(Av)
            e = wc * Ec + args.w_fit * Ef + args.w_will * W
            if use_bar:
                d = jnp.clip(pt_tri_dist(X[pi], X[Fi[pj]], X[Fj[pj]], X[Fk[pj]]), 1e-9, dhat)
                e = e + args.w_rep * jnp.sum(-(d - dhat) ** 2 * jnp.log(d / dhat)) / nV
            return e
        self.vg = jax.jit(jax.value_and_grad(energy))
        self.ptd = jax.jit(pt_tri_dist)
        self.contact = Contact(self.F, self.E, nV); self.idhat = args.ipc_dhat * self.h0
        self._ipc_parts = (unpack, conf_res, willmore, Fi, Fj, Fk, Ei, Ej)
        self.rebuild_ipc()

    def rebuild_ipc(self):
        unpack, conf_res, willmore, Fi, Fj, Fk, Ei, Ej = self._ipc_parts; idh = self.idhat
        Ea, Eb = Ei, Ej
        def bar(d, m):
            x = jnp.clip(d, 1e-12, idh) / idh
            return jnp.sum(m * (-(x - 1.0) ** 2 * jnp.log(x)))
        def energy_ipc(z, wc, tgt, nrm, pv, pf, pm, ea, eb, em):
            X, u = unpack(z); r = conf_res(X, u); Ec = jnp.mean(r ** 2); W, Av = willmore(X); dv = X - tgt
            Ef = jnp.sum(Av * (jnp.sum(dv * nrm, -1) ** 2 + 0.05 * jnp.sum(dv ** 2, -1))) / jnp.sum(Av)
            dpt = pt_tri_dist(X[pv], X[Fi[pf]], X[Fj[pf]], X[Fk[pf]])
            dee = ee_dist_j(X[Ea[ea]], X[Eb[ea]], X[Ea[eb]], X[Eb[eb]])
            return wc * Ec + args.w_fit * Ef + args.w_will * W + args.w_rep * wc * (bar(dpt, pm) + bar(dee, em))
        self.vg_ipc = jax.jit(jax.value_and_grad(energy_ipc))

    def pairs(self, X):
        F = self.F; cen = X[F].mean(1); rad = np.linalg.norm(X[F] - cen[:, None], axis=2).max(1)
        cand = cKDTree(cen).query_ball_point(X, self.dhat + rad.max()); vi, fj = [], []
        for v, fs in enumerate(cand):
            for f in fs:
                if v in F[f] or any(w in self.near[v] for w in F[f]): continue
                vi.append(v); fj.append(f)
        if vi:
            vi, fj = np.array(vi), np.array(fj)
            d = np.asarray(self.ptd(jnp.array(X[vi]), jnp.array(X[F[fj, 0]]), jnp.array(X[F[fj, 1]]), jnp.array(X[F[fj, 2]])))
            k = d < self.dhat
            if k.any(): return vi[k], fj[k]
        return np.zeros(1, int), np.zeros(1, int)   # dummy pair (vertex 0 of face 0): contributes 0 because d is clipped

    def metrics(self, X, u):
        r = np.asarray(self.conf_res(jnp.array(X), jnp.array(u))); _, _, dist = project_to_P(X, VP, F0)
        W, _ = self.willmore(jnp.array(X))
        return dict(V=self.nV, conf_rms=float(np.sqrt(np.mean(r ** 2))), conf_max=float(np.abs(r).max()),
                    dist_rms=float(np.sqrt(np.mean(dist ** 2))) * 300, dist_max=float(dist.max()) * 300,
                    willmore_over_4pi=float(W / (4 * np.pi)), self_intersections=self_intersections(X, self.F),
                    sym_err=float(np.abs(X[self.tau] - X @ T.T).max()))

def solve(L, X0, u0):
    z = np.concatenate([X0.ravel(), u0]); guard = args.barrier == 'on'
    if guard and self_intersections(X0, L.F) > 0:
        say('  initial state intersects itself; barrier guard disabled on this level'); guard = False
    t0 = time.time(); hist = []
    for wc in [float(x) for x in args.wc_schedule.split(',')]:
        burst, done_it, rejects, diag = 25, 0, 0, {}
        while done_it < args.iters:
            X, u = (np.asarray(a) for a in L.unpack(jnp.array(z)))
            tgt, nrm, _ = project_to_P(X, VP, F0)
            pi, pj = L.pairs(X) if args.barrier == 'on' else (np.zeros(1, int), np.zeros(1, int))
            a = (jnp.array(tgt), jnp.array(nrm), jnp.array(pi), jnp.array(pj))
            f = lambda zz: tuple(np.asarray(v, dtype=float) for v in L.vg(jnp.array(zz), wc, *a))
            res = minimize(f, z, jac=True, method='L-BFGS-B', options=dict(maxiter=burst, maxcor=30))
            znew, alpha = res.x, 1.0
            while guard:
                Xn, _ = (np.asarray(a_) for a_ in L.unpack(jnp.array(znew)))
                if self_intersections(Xn, L.F) == 0: break
                alpha /= 2; znew = z + alpha * (res.x - z)
                if alpha < 1 / 256: break
            if alpha < 1 / 256:
                rejects += 1; burst = max(3, burst // 2)
                if rejects > 30: break
                continue
            moved = np.abs(znew - z).max(); z = znew; done_it += max(1, res.nit)
            burst = min(50, burst + 5) if alpha == 1.0 else max(3, burst // 2)
            diag = dict(nit=int(res.nit), alpha=alpha, msg=str(res.message)[:24], active=int(len(pi)))
            if res.nit < 2 or moved < 1e-12: break
        X, u = (np.asarray(a) for a in L.unpack(jnp.array(z)))
        m = L.metrics(X, u); m.update(wc=wc, rejects=rejects, seconds=round(time.time() - t0, 1), diag=diag); hist.append(m)
        say(f'  L{L.M["level"]} wc={wc:<7g} conf_rms={m["conf_rms"]:.2e} conf_max={m["conf_max"]:.2e} dist_rms={m["dist_rms"]:.2f} '
            f'dist_max={m["dist_max"]:.2f} W/4pi={m["willmore_over_4pi"]:.3f} selfint={m["self_intersections"]} rej={rejects} t={m["seconds"]}s {diag}')
    return X, u, hist

def _pad(a, m=None):
    n = max(64, 1 << int(np.ceil(np.log2(max(1, len(a))))))
    out = np.zeros(n, int); out[:len(a)] = a; mask = np.zeros(n); mask[:len(a)] = 1.0
    return out, mask

def solve_ipc(L, X0, u0):
    """contact-aware L-BFGS: pairs recomputed at every energy evaluation, every step bounded by CCD"""
    C = L.contact; z = np.concatenate([X0.ravel(), u0])
    assert self_intersections(X0, L.F) == 0, 'IPC needs an embedded start'
    g0 = C.min_distance(X0, L.idhat * 4)
    if args.target == 'P' and g0 < 2 * L.idhat:   # start = P with its thin spikes: shrink dhat (target cp unchanged for reproducibility)
        L.idhat = 0.5 * g0; L.rebuild_ipc()
        say(f'  IPC barrier range reduced to {L.idhat * 300:.3f} units (half the initial gap {g0 * 300:.3f})')
    lin = lambda d: np.asarray(L.unpack(jnp.array(d))[0])         # z-direction -> vertex displacement (linear)
    t0 = time.time(); hist = []
    for wc in [float(x) for x in args.wc_schedule.split(',')]:
        X = lin(z); tgt, nrm, _ = project_to_P(X, VP, F0); jt, jn = jnp.array(tgt), jnp.array(nrm)
        def fg(zz):
            Xz = lin(zz)
            if not np.all(np.isfinite(Xz)): return np.inf, None
            pv, pf, ea, eb = C.pairs(Xz, L.idhat)
            pv, pm = _pad(pv); pf, _ = _pad(pf); ea, em = _pad(ea); eb, _ = _pad(eb)
            e, g = L.vg_ipc(jnp.array(zz), wc, jt, jn, jnp.array(pv), jnp.array(pf), jnp.array(pm), jnp.array(ea), jnp.array(eb), jnp.array(em))
            e = float(e)
            return (e, np.asarray(g)) if np.isfinite(e) else (np.inf, None)
        f, g = fg(z); S, Y = [], []; stall = 0; nit = 0; amin = 1.0
        for it in range(args.iters):
            q = g.copy(); al = []
            for s_, y_ in reversed(list(zip(S, Y))):
                a_ = (s_ @ q) / (y_ @ s_); al.append(a_); q -= a_ * y_
            if S: q *= (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
            for (s_, y_), a_ in zip(zip(S, Y), reversed(al)):
                b_ = (y_ @ q) / (y_ @ s_); q += (a_ - b_) * s_
            d = -q
            if g @ d >= 0: d = -g; S, Y = [], []
            DX = lin(d); m = np.linalg.norm(DX, axis=1).max()
            cap = min(1.0, L.h0 / m) if m > 0 else 1.0
            alpha = min(1.0, cap * C.ccd(lin(z), cap * DX))
            ok = False
            for _ in range(40):
                zn = z + alpha * d; fn, gn = fg(zn)
                if np.isfinite(fn) and fn <= f + 1e-4 * alpha * (g @ d): ok = True; break
                alpha /= 2
            if not ok: break
            sv, yv = zn - z, gn - g
            if sv @ yv > 1e-16 * (yv @ yv): S.append(sv); Y.append(yv); S, Y = S[-20:], Y[-20:]
            stall = stall + 1 if abs(f - fn) < 1e-13 * max(1.0, abs(f)) else 0
            z, f, g = zn, fn, gn; nit = it + 1; amin = min(amin, alpha)
            if stall >= 5: break
            if (it + 1) % 25 == 0:                       # refresh the fit targets (changes the energy -> reset memory)
                X = lin(z); tgt, nrm, _ = project_to_P(X, VP, F0); jt, jn = jnp.array(tgt), jnp.array(nrm); f, g = fg(z); S, Y = [], []
        X, u = (np.asarray(a) for a in L.unpack(jnp.array(z)))
        m = L.metrics(X, u); m.update(wc=wc, seconds=round(time.time() - t0, 1),
                                       diag=dict(nit=nit, min_alpha=float(amin), min_gap=float(C.min_distance(X, L.idhat * 4)) * 300)); hist.append(m)
        say(f'  L{L.M["level"]} wc={wc:<7g} conf_rms={m["conf_rms"]:.2e} conf_max={m["conf_max"]:.2e} dist_rms={m["dist_rms"]:.2f} '
            f'dist_max={m["dist_max"]:.2f} W/4pi={m["willmore_over_4pi"]:.3f} selfint={m["self_intersections"]} t={m["seconds"]}s {m["diag"]}')
    return X, u, hist

summary = dict(args=vars(args), levels={})
if args.resume and os.path.exists(f'{args.out}/{tag}.json'):
    summary['levels'] = {int(k): v for k, v in json.load(open(f'{args.out}/{tag}.json'))['levels'].items()}
rng = np.random.default_rng(args.seed); prev = None
for lev in [int(x) for x in args.levels.split(',')]:
    L = Level(lev)
    ck = f'{args.out}/{tag}_L{lev}.npz'
    if args.resume and lev in summary['levels'] and os.path.exists(ck):
        Z = np.load(ck); prev = (L, Z['X'] / 300.0, Z['u']); say(f'[{tag}] level {lev}: resumed from checkpoint'); continue
    if prev is None:
        G = rng.normal(size=L.XP.shape); G = np.asarray(L.unpack(jnp.array(np.concatenate([G.ravel(), np.zeros(L.nV)])))[0])
        amp = args.noise * L.h0
        while amp > 0 and self_intersections(L.XP + amp * G, L.F) > 0:
            amp /= 2                              # the polyhedron's narrowest gap is only 2.84 units
        X0 = L.XP + amp * G
        if args.noise > 0: say(f'[{tag}] initial perturbation amplitude {amp * 300:.3f} units (embedded)')
        l0 = np.linalg.norm(X0[L.E[:, 0]] - X0[L.E[:, 1]], axis=1)
        A = sp.coo_matrix((np.full(2 * len(L.E), 0.5), (np.r_[np.arange(len(L.E)), np.arange(len(L.E))], np.r_[L.E[:, 0], L.E[:, 1]])),
                          shape=(len(L.E), L.nV)).tocsr()
        u0 = spl.lsqr(A, np.log(l0) - L.logLhat)[0]
        lab = 'polyhedron + noise' if args.noise > 0 else 'polyhedron'
        say(f'[{tag}] level {lev}: V={L.nV}; initial state ({lab}): ' + json.dumps(L.metrics(X0, u0)))
    else:
        Lc, Xc, uc = prev; ia, ib = prolongation(Lc.M, L.M)
        X0 = (Xc[ia] + Xc[ib]) / 2; u0 = (uc[ia] + uc[ib]) / 2
        say(f'[{tag}] level {lev}: V={L.nV}; prolongated from level {Lc.M["level"]}')
    X, u, hist = (solve_ipc if args.solver == 'ipc' else solve)(L, X0, u0)
    entry = dict(history=hist, final=hist[-1])
    if prev is not None:   # geometric convergence: coarse vertices before vs after refinement
        Lc, Xc, _ = prev; ia, ib = prolongation(Lc.M, L.M); same = ia == ib
        entry['move_of_coarse_vertices_max'] = float(np.linalg.norm(X[same] - Xc[ia[same]], axis=1).max()) * 300
        say(f'  coarse vertices moved by at most {entry["move_of_coarse_vertices_max"]:.3f} units under refinement')
    summary['levels'][lev] = entry
    np.savez(ck, X=X * 300, u=u, F=L.F, XP=L.XP * 300, level=lev)
    json.dump(summary, open(f'{args.out}/{tag}.json', 'w'), indent=1)
    prev = (L, X, u)
say('done', tag)
