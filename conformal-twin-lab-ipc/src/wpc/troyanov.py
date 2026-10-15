"""Discrete Troyanov inverse: from the decorated hyperbolic surface (h_P on the fine mesh, the 24 marked
vertices, their cone angles) back to a flat cone metric, by solving for u with
    l_ij = exp((u_i + u_j)/2) * 2 sinh(L_ij / 2)      (Bobenko-Pinkall-Springborn)
such that the Euclidean angle sums are theta_i at the marked vertices and 2 pi elsewhere."""
import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
def flat_from_hyperbolic(M, L, marked, theta, verbose=False):
    F = M['F']; E = M['edges']; nV = len(M['XP'])
    eid = {(int(a), int(b)): k for k, (a, b) in enumerate(E)}; key = lambda a, b: eid[(min(a, b), max(a, b))]
    e_ab = np.array([key(a, b) for a, b, c in F]); e_bc = np.array([key(b, c) for a, b, c in F]); e_ca = np.array([key(c, a) for a, b, c in F])
    Fa, Fb, Fc = (jnp.array(F[:, k]) for k in range(3)); Ei, Ej = jnp.array(E[:, 0]), jnp.array(E[:, 1])
    lam = jnp.array(2 * np.sinh(L / 2))
    target = np.full(nV, 2 * np.pi); target[marked] = theta; target = jnp.array(target)
    def lengths(u): return jnp.exp(0.5 * (u[Ei] + u[Ej])) * lam
    def eang(a, b, c):
        # Euclidean angle opposite a; continuous extension to degenerate triangles (Bobenko-Pinkall-Springborn):
        # if a >= b + c the angle opposite a is pi, if b >= a + c or c >= a + b it is 0
        num = (a - b + c) * (a + b - c); den = (a + b + c) * (-a + b + c)
        valid = (a < b + c) & (b < a + c) & (c < a + b)
        ang = 2 * jnp.arctan(jnp.sqrt(jnp.where(valid, num, 1.0) / jnp.where(valid, den, 1.0)))
        return jnp.where(valid, ang, jnp.where(a >= b + c, jnp.pi, 0.0))
    def F_(u):
        l = lengths(u); lab, lbc, lca = l[e_ab], l[e_bc], l[e_ca]
        T = jnp.zeros(nV).at[Fa].add(eang(lbc, lab, lca)).at[Fb].add(eang(lca, lab, lbc)).at[Fc].add(eang(lab, lbc, lca))
        return T - target
    def ok(u):
        l = np.asarray(lengths(jnp.array(u))); a, b, c = l[e_ab], l[e_bc], l[e_ca]
        return bool(np.all(a < b + c) & np.all(b < a + c) & np.all(c < a + b))
    f = jax.jit(F_); J = jax.jit(jax.jacfwd(F_)); u = np.zeros(nV)
    start_ok = ok(u)
    for it in range(100):
        r = np.asarray(f(jnp.array(u))); res = np.abs(r).max()
        if res < 1e-12: break
        d = np.linalg.lstsq(np.asarray(J(jnp.array(u))), -r, rcond=None)[0]; d -= d.mean(); a = 1.0   # Euclidean: fix the global scale
        while a > 1e-8:
            un = u + a * d
            if np.abs(np.asarray(f(jnp.array(un)))).max() < res * (1 - 1e-4 * a): break
            a /= 2
        if a <= 1e-8: break
        u = un
    return np.asarray(lengths(jnp.array(u))), float(res), it, dict(start_valid=start_ok, end_valid=ok(u))
