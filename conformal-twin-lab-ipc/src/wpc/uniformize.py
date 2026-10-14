"""Discrete uniformisation of Mizhaev's polyhedron P: the hyperbolic metric h_P that is discretely
conformal (Bobenko-Pinkall-Springborn) to P's flat cone metric on the fine mesh:
    2 sinh(L_ij / 2) = exp((u_i + u_j)/2) * l_ij      with all angle sums = 2 pi.
Damped Newton on the angle-sum equations (Jacobian by JAX)."""
import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
def uniformize(M, verbose=True):
    F = M['F']; E = M['edges']; X = M['XP']; nV = len(X)
    ell = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
    eid = {(int(a), int(b)): k for k, (a, b) in enumerate(E)}
    key = lambda a, b: eid[(min(a, b), max(a, b))]
    e_ab = np.array([key(a, b) for a, b, c in F]); e_bc = np.array([key(b, c) for a, b, c in F]); e_ca = np.array([key(c, a) for a, b, c in F])
    Fa, Fb, Fc = (jnp.array(F[:, k]) for k in range(3)); Ei, Ej = jnp.array(E[:, 0]), jnp.array(E[:, 1]); jl = jnp.array(ell)
    def lengths(u): return 2 * jnp.arcsinh(jnp.exp(0.5 * (u[Ei] + u[Ej])) * jl / 2)
    def ang(a, b, c): return jnp.arccos(jnp.clip((jnp.cosh(b) * jnp.cosh(c) - jnp.cosh(a)) / (jnp.sinh(b) * jnp.sinh(c)), -1, 1))
    def theta(u):
        L = lengths(u); lab, lbc, lca = L[e_ab], L[e_bc], L[e_ca]
        T = jnp.zeros(nV)
        T = T.at[Fa].add(ang(lbc, lab, lca)).at[Fb].add(ang(lca, lab, lbc)).at[Fc].add(ang(lab, lbc, lca))
        return T - 2 * jnp.pi
    def tri_ok(u):
        L = np.asarray(lengths(u)); a, b, c = L[e_ab], L[e_bc], L[e_ca]
        return bool(np.all(a < b + c) & np.all(b < a + c) & np.all(c < a + b))
    th = jax.jit(theta); jac = jax.jit(jax.jacfwd(theta))
    Lmed = np.median(M['lenH']); u = np.full(nV, 2 * np.log(2 * np.sinh(Lmed / 2) / np.median(ell)))
    for it in range(100):
        r = np.asarray(th(jnp.array(u))); res = np.abs(r).max()
        if verbose and (it % 5 == 0 or res < 1e-12): print(f'    newton {it:3d}  max|angle sum - 2pi| = {res:.3e}')
        if res < 1e-12: break
        Jm = np.asarray(jac(jnp.array(u))); d = np.linalg.lstsq(Jm, -r, rcond=None)[0]; a = 1.0
        while a > 1e-8:
            un = u + a * d
            if tri_ok(un) and np.abs(np.asarray(th(jnp.array(un)))).max() < res * (1 - 1e-4 * a): break
            a /= 2
        if a <= 1e-8: print('    Newton stopped: no further decrease at double precision'); break
        u = un
    L = np.asarray(lengths(jnp.array(u)))
    return dict(u=u, L=L, residual=float(res), area=float(56 * 4 ** M['level'] * np.pi - np.pi * 2 * nV) , ok=tri_ok(u))
