"""Stretched Spectral Galerkin utilities.

This module is currently a cleaned extraction point for the SSG notebook.
The public entry point is expected to be `ssg_solve`.
"""

# =============================================================================
#  ZELLE 1 — SSG (Stretched Spectral Galerkin) für triaxialen Ellipsoid
#  Pullback auf Einheitskugel, Galerkin in Bessel-Sphärenharmonik-Basis.
#  Mass-Matrix = I (Basis ist L^2-orthonormal).
# =============================================================================
import numpy as np
import time

from scipy.special import spherical_jn
try:
    from scipy.special import sph_harm_y
except ImportError:
    from scipy.special import sph_harm

    def sph_harm_y(l, m, theta, phi):
        """
        Compatibility wrapper for older SciPy versions.

        New scipy.special.sph_harm_y uses:
            sph_harm_y(l, m, theta, phi)

        Old scipy.special.sph_harm uses:
            sph_harm(m, l, phi, theta)

        Here theta is the polar angle and phi is the azimuthal angle.
        """
        return sph_harm(m, l, phi, theta)

from scipy.optimize import brentq
from scipy.linalg import eigh

from hearing_ellipsoid_bench.core.types import AlgorithmResult, clean_eigenvalues

def mcmahon(l, n):
    nu = l + 0.5; mu = 4 * nu * nu
    beta = (n + l / 2) * np.pi
    return beta - (mu - 1) / (8 * beta)


def spherical_bessel_zeros(l, n_zeros):
    """First n_zeros positive zeros of j_l, ~14 digits."""
    zeros = np.empty(n_zeros)
    x_max = mcmahon(l, n_zeros) + 2 * np.pi
    x_min = max(1e-8, l * 0.9 + 0.1)
    n_grid = max(200, 25 * n_zeros)
    xs = np.linspace(x_min, x_max, n_grid)
    fs = spherical_jn(l, xs)
    s = np.sign(fs); s[s == 0] = 1
    sc = np.where(np.diff(s) != 0)[0]
    while len(sc) < n_zeros:
        x_max *= 2; n_grid *= 2
        xs = np.linspace(x_min, x_max, n_grid)
        fs = spherical_jn(l, xs)
        s = np.sign(fs); s[s == 0] = 1
        sc = np.where(np.diff(s) != 0)[0]
    f = lambda x: spherical_jn(l, x)
    for k in range(n_zeros):
        i = sc[k]
        zeros[k] = brentq(f, xs[i], xs[i + 1], xtol=1e-14, rtol=4e-15)
    return zeros


def build_basis(l_max, n_max_per_l):
    """Build (l, n, m) basis with Bessel zeros and L^2-normalizations."""
    basis = []
    alphas = {}; norms = {}
    for l in range(l_max + 1):
        zs = spherical_bessel_zeros(l, n_max_per_l)
        for n in range(1, n_max_per_l + 1):
            alpha = zs[n - 1]
            alphas[(l, n)] = alpha
            norms[(l, n)] = np.sqrt(2.0) / abs(spherical_jn(l + 1, alpha))
            for m in range(-l, l + 1):
                basis.append((l, n, m))
    return basis, alphas, norms


def build_quadrature(n_radial, n_theta, n_phi):
    """Product quadrature on B_1: GL radial, GL in cos(theta), trapez phi."""
    r_nodes_std, r_w_std = np.polynomial.legendre.leggauss(n_radial)
    r_q = 0.5 * (r_nodes_std + 1.0); r_w = 0.5 * r_w_std
    mu_q, mu_w = np.polynomial.legendre.leggauss(n_theta)
    theta_q = np.arccos(mu_q)
    sin_theta_q = np.sqrt(1.0 - mu_q ** 2)
    phi_q = np.arange(n_phi) * (2.0 * np.pi / n_phi)
    return {"r_q": r_q, "r_w": r_w, "mu_q": mu_q, "theta_q": theta_q,
            "sin_theta_q": sin_theta_q, "mu_w": mu_w, "phi_q": phi_q}


def tabulate_radial(basis, alphas, norms, r_q):
    table = {}
    done = set()
    for (l, n, m) in basis:
        if (l, n) in done: continue
        alpha = alphas[(l, n)]; N = norms[(l, n)]
        ar = alpha * r_q
        jl = spherical_jn(l, ar)
        R = N * jl
        if l == 0:
            jl_prime = -spherical_jn(1, ar)
        else:
            jl_prime = spherical_jn(l - 1, ar) - (l + 1) / ar * jl
        dRdr = N * alpha * jl_prime
        table[(l, n)] = (R, dRdr)
        done.add((l, n))
    return table


def tabulate_angular(basis, theta_q, phi_q, mu_q, sin_theta_q):
    """Y_lm and analytic derivatives on the angular grid."""
    table = {}
    done = set()
    T, P = np.meshgrid(theta_q, phi_q, indexing='ij')
    cot_theta = (mu_q[:, None] / sin_theta_q[:, None])
    for (l, n, m) in basis:
        if (l, m) in done: continue
        Y = sph_harm_y(l, m, T, P)
        dY_dphi = 1j * m * Y
        if m < l:
            Y_mp1 = sph_harm_y(l, m + 1, T, P)
            term2 = np.sqrt((l - m) * (l + m + 1)) * np.exp(-1j * P) * Y_mp1
        else:
            term2 = 0.0
        dY_dtheta = m * cot_theta * Y + term2
        table[(l, m)] = (Y, dY_dtheta, dY_dphi)
        done.add((l, m))
    return table


def assemble_stiffness(basis, radial_table, angular_table, quad,
                        a, b, c, verbose=True, block_size=None):
    """Assemble K_ij = (1/a^2)<u_x,v_x> + (1/b^2)<u_y,v_y> + (1/c^2)<u_z,v_z>.

    block_size: if not None, build gradient table in chunks of this many basis
                functions to control peak memory. None = build all at once.
    """
    r_q = quad["r_q"]; r_w = quad["r_w"]
    theta_q = quad["theta_q"]; mu_w = quad["mu_w"]
    sin_theta_q = quad["sin_theta_q"]; phi_q = quad["phi_q"]
    n_r = len(r_q); n_theta = len(theta_q); n_phi = len(phi_q)
    n_quad = n_r * n_theta * n_phi
    n_basis = len(basis)

    if verbose:
        gb = 3 * n_basis * n_quad * 16 / 1e9
        print(f"  basis={n_basis}, quad={n_quad}, grad table ~{gb:.2f} GB (complex)")

    phi_w_val = 2.0 * np.pi / n_phi
    W3 = (r_q[:, None, None] ** 2 * r_w[:, None, None]
          * mu_w[None, :, None] * phi_w_val
          * np.ones((1, 1, n_phi)))
    W = W3.reshape(-1)
    sqrtW = np.sqrt(W)

    # Cartesian basis vectors on angular grid
    sin_t = sin_theta_q[:, None]
    cos_t = quad["mu_q"][:, None]
    sin_p = np.sin(phi_q)[None, :]
    cos_p = np.cos(phi_q)[None, :]
    rhat_x = sin_t * cos_p; rhat_y = sin_t * sin_p; rhat_z = cos_t * np.ones_like(sin_p)
    that_x = cos_t * cos_p; that_y = cos_t * sin_p; that_z = -sin_t * np.ones_like(sin_p)
    phat_x = -sin_p * np.ones_like(sin_t); phat_y = cos_p * np.ones_like(sin_t)
    inv_r = 1.0 / r_q
    inv_sin_theta = 1.0 / sin_theta_q

    def build_grad_block(i0, i1):
        n = i1 - i0
        gx = np.empty((n, n_quad), dtype=np.complex128)
        gy = np.empty((n, n_quad), dtype=np.complex128)
        gz = np.empty((n, n_quad), dtype=np.complex128)
        for k, idx in enumerate(range(i0, i1)):
            l, nn, m = basis[idx]
            R, dRdr = radial_table[(l, nn)]
            Y, dYdt, dYdp = angular_table[(l, m)]
            T1 = dRdr[:, None, None] * Y[None, :, :]
            T2 = (R * inv_r)[:, None, None] * dYdt[None, :, :]
            T3 = (R * inv_r)[:, None, None] * (inv_sin_theta[None, :, None] * dYdp[None, :, :])
            gx[k] = (T1 * rhat_x[None, :, :] + T2 * that_x[None, :, :]
                     + T3 * phat_x[None, :, :]).reshape(-1) * sqrtW
            gy[k] = (T1 * rhat_y[None, :, :] + T2 * that_y[None, :, :]
                     + T3 * phat_y[None, :, :]).reshape(-1) * sqrtW
            gz[k] = (T1 * rhat_z[None, :, :] + T2 * that_z[None, :, :]).reshape(-1) * sqrtW
        return gx, gy, gz

    if block_size is None or block_size >= n_basis:
        if verbose:
            print(f"  building full gradient table ...", flush=True)
        t0 = time.perf_counter()
        Gx, Gy, Gz = build_grad_block(0, n_basis)
        if verbose: print(f"  grad table: {time.perf_counter()-t0:.1f}s")

        if verbose: print(f"  GEMM-ing K ...", flush=True)
        t0 = time.perf_counter()
        K = ((1.0 / a**2) * (Gx.conj() @ Gx.T)
             + (1.0 / b**2) * (Gy.conj() @ Gy.T)
             + (1.0 / c**2) * (Gz.conj() @ Gz.T))
        if verbose: print(f"  GEMM: {time.perf_counter()-t0:.1f}s")
    else:
        if verbose:
            print(f"  block-mode: {block_size} basis funcs at a time", flush=True)
        K = np.zeros((n_basis, n_basis), dtype=np.complex128)
        # Build all blocks once, store (memory-mapped or full) -- needed for off-diagonal blocks.
        # For simplicity here: build full table after all (block_size only saves memory if
        # we accept rebuilding gradients -- the cleanest no-rebuild block-loop is below).
        # Rebuild-once strategy: build i-th column block on demand, j-th row block on demand,
        # but that's 2x the gradient work. Acceptable trade-off.
        n_blocks = (n_basis + block_size - 1) // block_size
        t0 = time.perf_counter()
        # Pre-build all blocks (still uses memory; if too much, rewrite as nested loop with rebuilding)
        blocks_x = []; blocks_y = []; blocks_z = []
        for bi in range(n_blocks):
            i0 = bi * block_size
            i1 = min(i0 + block_size, n_basis)
            gx, gy, gz = build_grad_block(i0, i1)
            blocks_x.append(gx); blocks_y.append(gy); blocks_z.append(gz)
            if verbose:
                print(f"    grad block {bi+1}/{n_blocks} ({i0}..{i1})", flush=True)
        if verbose: print(f"  total grad table: {time.perf_counter()-t0:.1f}s")

        if verbose: print(f"  GEMM-ing K block-wise ...", flush=True)
        t0 = time.perf_counter()
        for bi in range(n_blocks):
            i0 = bi * block_size; i1 = min(i0 + block_size, n_basis)
            for bj in range(bi, n_blocks):
                j0 = bj * block_size; j1 = min(j0 + block_size, n_basis)
                Kij = ((1.0 / a**2) * (blocks_x[bi].conj() @ blocks_x[bj].T)
                       + (1.0 / b**2) * (blocks_y[bi].conj() @ blocks_y[bj].T)
                       + (1.0 / c**2) * (blocks_z[bi].conj() @ blocks_z[bj].T))
                K[i0:i1, j0:j1] = Kij
                if bi != bj:
                    K[j0:j1, i0:i1] = Kij.conj().T
        if verbose: print(f"  GEMM: {time.perf_counter()-t0:.1f}s")

    K = 0.5 * (K + K.conj().T)
    return K


def ssg_solve(a, b, c, l_max, n_max, n_radial, n_theta, n_phi,
              n_eigs=None, block_size=None, verbose=True):
    """Top-level driver. Returns (eigs, basis, K)."""
    t0 = time.perf_counter()
    if verbose:
        print(f"=== SSG  a={a}, b={b}, c={c} ===")
        print(f"  basis l_max={l_max}, n_max={n_max}; quad ({n_radial},{n_theta},{n_phi})")
    basis, alphas, norms = build_basis(l_max, n_max)
    quad = build_quadrature(n_radial, n_theta, n_phi)
    rt = tabulate_radial(basis, alphas, norms, quad["r_q"])
    at = tabulate_angular(basis, quad["theta_q"], quad["phi_q"],
                           quad["mu_q"], quad["sin_theta_q"])
    K = assemble_stiffness(basis, rt, at, quad, a, b, c,
                            verbose=verbose, block_size=block_size)
    if n_eigs is None or n_eigs >= K.shape[0]:
        if verbose:
            print(f"  dense eigvalsh size {K.shape[0]} ...", flush=True)
        ts = time.perf_counter()
        eigs = np.linalg.eigvalsh(K)
        if verbose:
            print(f"  solver: {time.perf_counter()-ts:.1f}s")
    else:
        if verbose:
            print(f"  dense Hermitian partial eigensolve size {K.shape[0]}, n_eigs={n_eigs} ...", flush=True)
        ts = time.perf_counter()
        eigs = eigh(
            K,
            eigvals_only=True,
            subset_by_index=[0, n_eigs - 1],
            driver="evr",
            check_finite=False,
        )
        if verbose:
            print(f"  solver: {time.perf_counter()-ts:.1f}s")
    eigs = np.sort(eigs.real)
    if verbose: print(f"  TOTAL: {time.perf_counter()-t0:.1f}s")
    return eigs, basis, K

def solve_ssg(
    a: float,
    b: float,
    c: float,
    l_max: int = 22,
    n_max: int = 13,
    n_radial: int = 40,
    n_theta: int = 32,
    n_phi: int = 64,
    n_eigs=None,
    block_size=None,
    verbose: bool = True,
) -> "AlgorithmResult":
    """SSG-Solver im AlgorithmResult-Vertrag."""
    from hearing_ellipsoid_bench.core.types import AlgorithmResult, clean_eigenvalues
    import time

    name = "ssg_galerkin"
    t0 = time.perf_counter()

    try:
        eigs, basis, extra = ssg_solve(
            a=a, b=b, c=c,
            l_max=l_max,
            n_max=n_max,
            n_radial=n_radial,
            n_theta=n_theta,
            n_phi=n_phi,
            n_eigs=n_eigs,
            block_size=block_size,
            verbose=verbose,
        )

        eigs = clean_eigenvalues(eigs)

        return AlgorithmResult(
            algorithm=name,
            eigs=eigs,
            runtime_sec=time.perf_counter() - t0,
            success=True,
            meta={
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "l_max": int(l_max),
                "n_max": int(n_max),
                "n_radial": int(n_radial),
                "n_theta": int(n_theta),
                "n_phi": int(n_phi),
                "n_basis": int(len(basis)),
                "n_eigs_requested": None if n_eigs is None else int(n_eigs),
                "n_eigs_found": int(len(eigs)),
                "block_size": block_size,
                "extra": extra if isinstance(extra, dict) else repr(extra),
            },
        )

    except Exception as e:
        return AlgorithmResult(
            algorithm=name,
            eigs=clean_eigenvalues([]),
            runtime_sec=time.perf_counter() - t0,
            success=False,
            message=repr(e),
            meta={
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "l_max": int(l_max),
                "n_max": int(n_max),
                "n_radial": int(n_radial),
                "n_theta": int(n_theta),
                "n_phi": int(n_phi),
                "n_eigs_requested": None if n_eigs is None else int(n_eigs),
                "block_size": block_size,
            },
        )