from __future__ import annotations

import time
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.sparse.linalg import splu, eigsh
from scipy.linalg import cholesky, solve_triangular, svd


def estimate_lambda_max(Kmat, Mmat) -> float:
    vals = eigsh(Kmat, k=1, M=Mmat, which="LA", return_eigenvectors=False)
    return float(vals[0])


def make_m_orthonormal_probes(M, n_probes=7, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((M.shape[0], n_probes))
    G = Z.T @ (M @ Z)
    R = cholesky(G, lower=False)
    return solve_triangular(R, Z.T, lower=False).T


def chebyshev_block_moments(Kmat, Mmat, n_moments, lambda_max, lambda_min=0.0, n_probes=7, seed=42, verbose=True):
    c_glob = 0.5 * (lambda_max + lambda_min)
    e_glob = 0.5 * (lambda_max - lambda_min)
    M_lu = splu(Mmat.tocsc())
    Phi = make_m_orthonormal_probes(Mmat, n_probes=n_probes, seed=seed)
    MPhi = Mmat @ Phi
    G = np.empty((n_moments, n_probes, n_probes), dtype=float)
    V_prev = Phi.copy()
    G[0] = MPhi.T @ V_prev
    V_curr = (M_lu.solve(Kmat @ Phi) - c_glob * Phi) / e_glob
    G[1] = MPhi.T @ V_curr
    for k in range(2, n_moments):
        BV = (M_lu.solve(Kmat @ V_curr) - c_glob * V_curr) / e_glob
        V_next = 2.0 * BV - V_prev
        G[k] = MPhi.T @ V_next
        V_prev, V_curr = V_curr, V_next
        if verbose and (k + 1) % max(n_moments // 10, 1) == 0:
            print(f"    {k+1}/{n_moments}", flush=True)
    return G, c_glob, e_glob

def resolvent_chebyshev_block_moments(
    Kmat,
    Mmat,
    n_moments,
    sigma=100.0,
    n_probes=11,
    seed=42,
    verbose=True,
    mu_headroom=1.0,
    checkpoint_path=None,
    checkpoint_every=2000,
):
    """
    Chebyshev block moments for the resolvent-transformed operator

        R_sigma = (K + sigma M)^(-1) M,

    whose eigenvalues satisfy

        mu_i = 1 / (lambda_i + sigma).

    Since lambda_i > 0, the spectrum of R_sigma lies in (0, 1/sigma).

    checkpoint_path: optional .npz path. The three-term recursion state
    (V_prev, V_curr, the moments computed so far and the defining
    parameters) is saved every checkpoint_every steps and on completion;
    if the file exists and matches (sigma, seed, n_probes, mu_headroom,
    problem size), the recursion RESUMES instead of restarting. This
    makes long moment runs robust against walltime kills: a resubmitted
    SLURM task continues where the previous one stopped.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    A = (Kmat + sigma * Mmat).tocsc()

    if verbose:
        print(
            f"Factorizing K + sigma M with sigma={sigma:g}",
            flush=True,
        )

    A_lu = splu(A)

    Phi = make_m_orthonormal_probes(
        Mmat,
        n_probes=n_probes,
        seed=seed,
    )
    MPhi = Mmat @ Phi

    # Safe global interval for mu:
    #     0 < mu_i < 1/sigma.
    # mu_headroom > 1 places lambda = 0 strictly inside the Chebyshev
    # interval, so bandpass windows can extend below the physical
    # spectrum bottom without touching the interval endpoint (endpoint-
    # adjacent filter edges create rank-K edge artifacts in the pencil).
    mu_min = 0.0
    mu_max = mu_headroom / sigma

    c_glob = 0.5 * (mu_max + mu_min)
    e_glob = 0.5 * (mu_max - mu_min)

    G = np.empty(
        (n_moments, n_probes, n_probes),
        dtype=float,
    )

    def apply_scaled_resolvent(X):
        RX = A_lu.solve(Mmat @ X)
        return (RX - c_glob * X) / e_glob

    def _ckpt_meta():
        return np.array([sigma, float(seed), float(n_probes),
                         mu_headroom, float(Phi.shape[0])])

    def _save_ckpt(k_done, V_prev, V_curr):
        if checkpoint_path is None:
            return
        tmp = str(checkpoint_path) + ".tmp"
        np.savez(tmp, meta=_ckpt_meta(), k_done=k_done,
                 G=G[:k_done], V_prev=V_prev, V_curr=V_curr)
        Path(tmp + ".npz").replace(checkpoint_path)  # atomic on POSIX

    start_k = 2
    V_prev = V_curr = None
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        try:
            ck = np.load(checkpoint_path)
            if (np.allclose(ck["meta"], _ckpt_meta())
                    and int(ck["k_done"]) >= 2
                    and int(ck["k_done"]) <= n_moments):
                kd = int(ck["k_done"])
                G[:kd] = ck["G"]
                V_prev = ck["V_prev"]
                V_curr = ck["V_curr"]
                start_k = kd
                if verbose:
                    print(f"    resuming from checkpoint at {kd}/{n_moments}",
                          flush=True)
        except Exception as exc:  # corrupt checkpoint: restart cleanly
            if verbose:
                print(f"    checkpoint unusable ({exc}); restarting",
                      flush=True)
            V_prev = V_curr = None
            start_k = 2

    if V_prev is None:
        V_prev = Phi.copy()
        G[0] = MPhi.T @ V_prev
        V_curr = apply_scaled_resolvent(Phi)
        G[1] = MPhi.T @ V_curr

    for k in range(start_k, n_moments):
        BV = apply_scaled_resolvent(V_curr)
        V_next = 2.0 * BV - V_prev

        G[k] = MPhi.T @ V_next

        V_prev, V_curr = V_curr, V_next

        if checkpoint_path is not None and (k + 1) % checkpoint_every == 0:
            _save_ckpt(k + 1, V_prev, V_curr)
        if verbose and (k + 1) % max(n_moments // 10, 1) == 0:
            print(f"    {k + 1}/{n_moments}", flush=True)

    _save_ckpt(n_moments, V_prev, V_curr)
    return G, c_glob, e_glob

def windowed_resolvent_block_fdm(
    Kmat,
    Mmat,
    lambda_top,
    sigma=100.0,
    lambda_bottom=0.0,
    n_moments=3000,
    n_windows=12,
    n_probes=11,
    hankel_blocks=110,
    transition_frac=0.12,
    window_overlap=0.20,
    seed=42,
    rank_tol=1e-9,
    max_rank=450,
    verbose=True,
):
    """
    Windowed block-Hankel filter diagonalization in resolvent space.

    The transformation is

        mu = 1 / (lambda + sigma),

    and recovered mu-values are mapped back through

        lambda = 1 / mu - sigma.
    """
    t0 = time.perf_counter()

    G, c_glob, e_glob = resolvent_chebyshev_block_moments(
        Kmat,
        Mmat,
        n_moments=n_moments,
        sigma=sigma,
        n_probes=n_probes,
        seed=seed,
        verbose=verbose,
    )

    edges = np.linspace(
        lambda_bottom,
        lambda_top,
        n_windows + 1,
    )

    all_eigs = []
    win_info = []

    for w_idx in range(n_windows):
        lam_a = float(edges[w_idx])
        lam_b = float(edges[w_idx + 1])

        width = lam_b - lam_a

        lam_a_pad = max(
            lambda_bottom,
            lam_a - window_overlap * width,
        )
        lam_b_pad = min(
            lambda_top,
            lam_b + window_overlap * width,
        )

        # The resolvent reverses the order:
        # lambda_a < lambda_b  =>  mu_a > mu_b
        mu_low = 1.0 / (lam_b_pad + sigma)
        mu_high = 1.0 / (lam_a_pad + sigma)

        Gf, (theta_a, theta_b) = bandpass_block_moments(
            G,
            c_glob,
            e_glob,
            mu_low,
            mu_high,
            transition_frac=transition_frac,
        )

        mu_vals, info = block_fdm_pencil_filtered(
            Gf,
            c_glob,
            e_glob,
            theta_a,
            theta_b,
            hankel_blocks=hankel_blocks,
            rank_tol=rank_tol,
            max_rank=max_rank,
        )

        mu_vals = np.asarray(mu_vals, dtype=float)
        mu_vals = mu_vals[
            np.isfinite(mu_vals) & (mu_vals > 0)
        ]

        lam_vals = 1.0 / mu_vals - sigma
        lam_vals = lam_vals[
            np.isfinite(lam_vals)
            & (lam_vals >= lam_a)
            & (lam_vals <= lam_b)
        ]

        all_eigs.extend(lam_vals.tolist())

        info.update({
            "lambda_window": (lam_a, lam_b),
            "mu_window": (mu_low, mu_high),
            "n_in_keep": int(len(lam_vals)),
        })
        win_info.append(info)

        if verbose:
            print(
                f"window {w_idx + 1}/{n_windows}: "
                f"lambda=[{lam_a:.3f}, {lam_b:.3f}], "
                f"kept={len(lam_vals)}",
                flush=True,
            )

    vals = np.array(sorted(all_eigs), dtype=float)

    if len(vals) > 1:
        keep = [0]

        for i in range(1, len(vals)):
            rel_gap = (
                vals[i] - vals[keep[-1]]
            ) / max(abs(vals[keep[-1]]), 1e-12)

            if rel_gap > 1e-6:
                keep.append(i)

        vals = vals[keep]

    return vals, {
        "algorithm": "resolvent_multi_probe_block_hankel",
        "wall_seconds": time.perf_counter() - t0,
        "sigma": sigma,
        "lambda_bottom": lambda_bottom,
        "lambda_top": lambda_top,
        "n_moments": n_moments,
        "n_windows": n_windows,
        "n_probes": n_probes,
        "n_eigs_total": int(len(vals)),
        "windows": win_info,
    }

def bandpass_block_moments(G, c_glob, e_glob, lam_a, lam_b, transition_frac=0.15, n_grid=None, theta_chunk=1024):
    L, Kprobe, _ = G.shape
    if n_grid is None:
        n_grid = max(8 * L, 8192)
    x_a = np.clip((lam_a - c_glob) / e_glob, -1.0, 1.0)
    x_b = np.clip((lam_b - c_glob) / e_glob, -1.0, 1.0)
    theta_a = np.arccos(x_b)
    theta_b = np.arccos(x_a)
    width = max(theta_b - theta_a, 1e-15)
    taper = transition_frac * width
    G_filt = np.zeros_like(G)
    n_idx = np.arange(1, L)
    for start in range(0, n_grid, theta_chunk):
        end = min(start + theta_chunk, n_grid)
        q = np.arange(start, end)
        thetas = (q + 0.5) * np.pi / n_grid
        W = np.zeros_like(thetas)
        core = (thetas >= theta_a + taper) & (thetas <= theta_b - taper)
        W[core] = 1.0
        if taper > 0:
            left = (thetas >= theta_a) & (thetas < theta_a + taper)
            W[left] = 0.5 * (1.0 - np.cos(np.pi * (thetas[left] - theta_a) / taper))
            right = (thetas > theta_b - taper) & (thetas <= theta_b)
            W[right] = 0.5 * (1.0 - np.cos(np.pi * (theta_b - thetas[right]) / taper))
        if np.all(W == 0):
            continue
        C = np.cos(np.outer(n_idx, thetas))
        g = G[0][None, :, :] + 2.0 * np.einsum("nq,nij->qij", C, G[1:], optimize=True)
        gW = g * W[:, None, None]
        G_filt[0] += np.sum(gW, axis=0) / n_grid
        G_filt[1:] += 2.0 * np.einsum("nq,qij->nij", C, gW, optimize=True) / n_grid
    return G_filt, (theta_a, theta_b)


def build_block_hankel_pair(G_filt, H: int):
    L, Kprobe, _ = G_filt.shape
    if 2 * H >= L:
        raise ValueError(f"Need 2H < n_moments. Got H={H}, L={L}")
    H0 = np.zeros((H * Kprobe, H * Kprobe), dtype=G_filt.dtype)
    H1 = np.zeros_like(H0)
    for i in range(H):
        for j in range(H):
            H0[i*Kprobe:(i+1)*Kprobe, j*Kprobe:(j+1)*Kprobe] = G_filt[i + j]
            H1[i*Kprobe:(i+1)*Kprobe, j*Kprobe:(j+1)*Kprobe] = G_filt[i + j + 1]
    return H0, H1


def block_fdm_pencil_filtered(G_filt, c_glob, e_glob, theta_a, theta_b, hankel_blocks=100, rank_tol=1e-9, max_rank=350, angle_tol=1e-4):
    H0, H1 = build_block_hankel_pair(G_filt, hankel_blocks)
    U, s, Vt = svd(H0, full_matrices=False)
    rank_auto = int(np.sum(s > s[0] * rank_tol))
    rank = max(1, min(rank_auto, max_rank, len(s)))
    Ur, Sr, Vtr = U[:, :rank], s[:rank], Vt[:rank, :]
    T = (Ur.T @ H1 @ Vtr.T) / Sr[None, :]
    z = np.linalg.eigvals(T)
    on_circle = np.abs(np.abs(z) - 1.0) < 0.08
    alpha = np.angle(z)
    alpha = np.mod(alpha, 2*np.pi)
    alpha = np.where(alpha > np.pi, 2*np.pi - alpha, alpha)
    keep = on_circle & (alpha > theta_a - 1e-3) & (alpha < theta_b + 1e-3) & np.isfinite(alpha)
    alphas = np.sort(alpha[keep])
    if len(alphas) == 0:
        return np.array([]), {"rank": rank, "rank_auto": rank_auto, "n_on_circle": int(on_circle.sum()), "n_final": 0}
    idx = [0]
    for i in range(1, len(alphas)):
        if alphas[i] - alphas[idx[-1]] > angle_tol:
            idx.append(i)
    lambdas = np.sort(c_glob + e_glob * np.cos(alphas[idx]))
    return lambdas, {"rank": rank, "rank_auto": rank_auto, "n_on_circle": int(on_circle.sum()), "n_final": len(lambdas), "singular_values": s}


def windowed_block_fdm(Kmat, Mmat, lambda_top, n_moments=4000, n_windows=8, n_probes=7, lambda_max=None, hankel_blocks=100, transition_frac=0.15, window_overlap=0.15, seed=42, rank_tol=1e-9, max_rank=350, verbose=True):
    t0 = time.perf_counter()
    if lambda_max is None:
        lambda_max = 1.05 * estimate_lambda_max(Kmat, Mmat)
    G, c_glob, e_glob = chebyshev_block_moments(Kmat, Mmat, n_moments, lambda_max, n_probes=n_probes, seed=seed, verbose=verbose)
    edges = np.linspace(0.0, lambda_top, n_windows + 1)
    all_eigs, win_info = [], []
    for w_idx in range(n_windows):
        a, b = edges[w_idx], edges[w_idx + 1]
        width = b - a
        a_pad = max(0.0, a - window_overlap * width)
        b_pad = min(lambda_max, b + window_overlap * width)
        Gf, (ta, tb) = bandpass_block_moments(G, c_glob, e_glob, a_pad, b_pad, transition_frac=transition_frac)
        eigs, info = block_fdm_pencil_filtered(Gf, c_glob, e_glob, ta, tb, hankel_blocks=hankel_blocks, rank_tol=rank_tol, max_rank=max_rank)
        eigs = eigs[(eigs >= a) & (eigs <= b)]
        all_eigs.extend(eigs.tolist())
        info.update({"window": (float(a), float(b)), "n_in_keep": int(len(eigs))})
        win_info.append(info)
    vals = np.array(sorted(all_eigs))
    if len(vals) > 1:
        keep = [0]
        for i in range(1, len(vals)):
            rel_gap = (vals[i] - vals[keep[-1]]) / max(abs(vals[keep[-1]]), 1e-12)
            if rel_gap > 1e-5:
                keep.append(i)
        vals = vals[keep]
    return vals, {
        "wall_seconds": time.perf_counter() - t0,
        "n_moments": n_moments,
        "n_windows": n_windows,
        "n_probes": n_probes,
        "n_eigs_total": len(vals),
        "windows": win_info,
        "lambda_max_used": lambda_max,
        "lambda_top": lambda_top,
    }


def cluster_values_by_relative_gap(values, rel_gap=5e-4) -> pd.DataFrame:
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0:
        return pd.DataFrame(columns=["lambda_center", "count", "lambda_min", "lambda_max"])
    clusters, current = [], [values[0]]
    for x in values[1:]:
        if (x - current[-1]) / max(abs(current[-1]), 1e-15) <= rel_gap:
            current.append(x)
        else:
            arr = np.array(current)
            clusters.append({"lambda_center": float(np.median(arr)), "count": len(arr), "lambda_min": float(arr.min()), "lambda_max": float(arr.max())})
            current = [x]
    arr = np.array(current)
    clusters.append({"lambda_center": float(np.median(arr)), "count": len(arr), "lambda_min": float(arr.min()), "lambda_max": float(arr.max())})
    return pd.DataFrame(clusters)


def stable_pole_filter(list_of_eig_arrays, min_count=3, rel_gap=5e-4) -> pd.DataFrame:
    arrays = [np.asarray(x, dtype=float) for x in list_of_eig_arrays if len(x)]
    if not arrays:
        return pd.DataFrame(columns=["lambda_center", "count", "lambda_min", "lambda_max"])
    dfc = cluster_values_by_relative_gap(np.concatenate(arrays), rel_gap=rel_gap)
    return dfc[dfc["count"] >= min_count].copy().reset_index(drop=True)


# ===================================================================
# Gap-free tile-parallel resolvent block-Hankel extension
# ===================================================================
#
# Design:
#   * The target interval [0, lambda_top] is partitioned into DISJOINT
#     half-open tiles [t_k, t_{k+1}) (last tile closed) whose edges are
#     placed by inverting the three-term Weyl counting function, so each
#     tile is expected to contain the same number of eigenvalues.
#     Because the tiles partition the interval exactly, the union of the
#     per-tile results is gap-free and duplicate-free BY CONSTRUCTION;
#     completeness inside each tile is then audited against Weyl counts.
#   * Tiles are grouped into a small number of contiguous BANDS. Each
#     band gets its own resolvent shift sigma_band = sqrt(la_pad*lb_pad)
#     (the geometric mean maximizes the fraction of the mu-interval
#     (0, 1/sigma] occupied by the band) and ONE Chebyshev moment
#     sequence / ONE sparse LU factorization. All tiles of the band are
#     extracted from that single moment sequence via bandpass filters.
#     Parallelism is over (band, seed) pairs, e.g. as a SLURM array.
#   * The pencil extraction preserves multiplicity: pencil eigenvalues
#     are clustered with a tolerance far below the local mean level
#     spacing, and the cluster size (capped by the number of probes) is
#     reported as the multiplicity estimate. Multiplicities are
#     validated across independent probe seeds in the merge step.
# -------------------------------------------------------------------


def weyl_counting_function(lam, volume, surface, mean_curvature_integral=0.0):
    """Three-term Weyl estimate N(lambda) for the Dirichlet Laplacian.

    N(lambda) = V/(6 pi^2) lambda^{3/2} - S/(16 pi) lambda
                + C/(6 pi^2) lambda^{1/2},

    with C the integrated mean curvature for Hbar = (k1+k2)/2 (matching
    hearing_ellipsoid_bench.validation.weyl.WeylFit3).
    """
    lam = np.asarray(lam, dtype=float)
    A0 = volume / (6.0 * np.pi**2)
    A1 = -surface / (16.0 * np.pi)
    A2 = mean_curvature_integral / (6.0 * np.pi**2)
    return A0 * np.maximum(lam, 0.0) ** 1.5 + A1 * lam + A2 * np.sqrt(np.maximum(lam, 0.0))


def weyl_density(lam, volume, surface, mean_curvature_integral=0.0):
    """d N / d lambda of the three-term Weyl estimate."""
    lam = float(lam)
    A0 = volume / (6.0 * np.pi**2)
    A1 = -surface / (16.0 * np.pi)
    A2 = mean_curvature_integral / (6.0 * np.pi**2)
    return 1.5 * A0 * np.sqrt(max(lam, 1e-12)) + A1 + 0.5 * A2 / np.sqrt(max(lam, 1e-12))


def _invert_weyl(n_target, volume, surface, mean_curvature_integral=0.0, lam_hi=1e7):
    from scipy.optimize import brentq

    f = lambda l: weyl_counting_function(l, volume, surface, mean_curvature_integral) - n_target
    return float(brentq(f, 1e-9, lam_hi, xtol=1e-10, rtol=1e-14))


def weyl_equal_count_edges(
    n_tiles,
    volume,
    surface,
    mean_curvature_integral=0.0,
    lambda_top=None,
    n_target=None,
):
    """Tile edges t_0=0 < t_1 < ... < t_W so that each tile
    [t_k, t_{k+1}) is expected to contain N_total/W eigenvalues.

    Exactly one of lambda_top / n_target must be given.
    Returns (edges, n_total_expected).
    """
    if (lambda_top is None) == (n_target is None):
        raise ValueError("give exactly one of lambda_top or n_target")
    if lambda_top is None:
        lambda_top = _invert_weyl(float(n_target), volume, surface, mean_curvature_integral)
    n_total = float(weyl_counting_function(lambda_top, volume, surface, mean_curvature_integral))
    edges = [0.0]
    for k in range(1, n_tiles):
        edges.append(_invert_weyl(n_total * k / n_tiles, volume, surface,
                                  mean_curvature_integral, lam_hi=2.0 * lambda_top + 10.0))
    edges.append(float(lambda_top))
    return np.asarray(edges, dtype=float), n_total


def group_tiles_into_bands(edges, n_bands):
    """Group W tiles into n_bands contiguous bands of (nearly) equal
    tile count. Returns list of (tile_lo, tile_hi_exclusive) pairs."""
    n_tiles = len(edges) - 1
    if not (1 <= n_bands <= n_tiles):
        raise ValueError("need 1 <= n_bands <= n_tiles")
    base, extra = divmod(n_tiles, n_bands)
    bands, start = [], 0
    for b in range(n_bands):
        size = base + (1 if b < extra else 0)
        bands.append((start, start + size))
        start += size
    return bands


def optimal_band_sigma(lam_a, lam_b, sigma_floor=5.0, low_frac=0.05):
    """Shift maximizing the mu-interval fraction occupied by [lam_a, lam_b]:
    the fraction f(sigma) = sigma (b-a) / ((a+sigma)(b+sigma)) is maximal
    at sigma = sqrt(a b); a floor keeps sigma sane for the lowest band."""
    a_eff = max(float(lam_a), low_frac * float(lam_b))
    return float(max(sigma_floor, np.sqrt(a_eff * float(lam_b))))


def auto_moments_for_band(
    lam_a,
    lam_b,
    sigma,
    volume,
    surface,
    mean_curvature_integral=0.0,
    resolution_factor=1.2,
    l_min=1500,
    l_max=20000,
    round_to=250,
):
    """Choose the Chebyshev moment count L for a band.

    The bandpass/pencil works in the Chebyshev angle theta of the
    resolvent variable mu = 1/(lambda+sigma); the Fourier limit of L
    moments is pi/L in theta. The matrix pencil super-resolves beyond
    that, so we require pi/L <= resolution_factor * s_theta, where
    s_theta is the smallest mean level spacing (mapped to theta) inside
    the band. Spacing in theta:  s_theta = s_lambda / (dlambda/dtheta),
    dlambda/dtheta = e sin(theta) / mu^2 with c = e = 1/(2 sigma).
    """
    e = 1.0 / (2.0 * sigma)
    s_theta_min = np.inf
    for lam in np.linspace(max(lam_a, 1e-6), lam_b, 33):
        mu = 1.0 / (lam + sigma)
        x = mu / e - 1.0
        sin_t = np.sqrt(max(1.0 - min(x * x, 1.0), 1e-12))
        dlam_dtheta = e * sin_t / mu**2
        rho = weyl_density(lam, volume, surface, mean_curvature_integral)
        s_lam = 1.0 / max(rho, 1e-12)
        s_theta_min = min(s_theta_min, s_lam / dlam_dtheta)
    L = np.pi / (resolution_factor * s_theta_min)
    L = int(np.clip(np.ceil(L / round_to) * round_to, l_min, l_max))
    return L


def _cluster_sorted(values, tol, cap=None):
    """Cluster a sorted 1-D array by gap tolerance.
    Returns (centers, counts)."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.array([]), np.array([], dtype=int)
    centers, counts = [], []
    cluster = [values[0]]
    for v in values[1:]:
        if v - cluster[-1] <= tol:
            cluster.append(v)
        else:
            centers.append(float(np.median(cluster)))
            counts.append(len(cluster) if cap is None else min(len(cluster), cap))
            cluster = [v]
    centers.append(float(np.median(cluster)))
    counts.append(len(cluster) if cap is None else min(len(cluster), cap))
    return np.asarray(centers), np.asarray(counts, dtype=int)


def _decimated_pencil_angles(
    G_filt,
    D,
    hankel_dim_max=6000,
    rank_tol=1e-10,
    max_rank=500,
    circle_tol=0.08,
):
    """Pencil on the D-decimated filtered moments.

    The analytically filtered moments form a one-sided exponential sum
        z_k ~ sum_j w_j W(theta_j) e^{i k theta_j}.
    Decimation by D maps theta_j -> D * theta_j (mod 2 pi): the narrow
    filter band is spread over the circle, so a small pencil built
    from moments Z[0], Z[D], ..., Z[2H*D] uses the information of
    ~2*H*D of the L computed moments instead of only the first 2H.
    Angle noise is also divided by D on decoding.

    Returns decimated angles in (-pi, pi], one per level.
    """
    Gd = np.ascontiguousarray(G_filt[::max(1, int(D))])
    Kprobe = Gd.shape[1]
    H = min(hankel_dim_max // Kprobe, (len(Gd) - 1) // 2)
    if H < 2:
        return np.array([]), {"hankel_blocks_used": int(H), "rank": 0,
                              "rank_auto": 0, "n_on_circle": 0}
    H0, H1 = build_block_hankel_pair(Gd, H)
    U, s, Vt = svd(H0, full_matrices=False)
    rank_auto = int(np.sum(s > s[0] * rank_tol)) if s[0] > 0 else 0
    rank = max(1, min(rank_auto, max_rank, len(s)))
    Ur, Sr, Vtr = U[:, :rank], s[:rank], Vt[:rank, :]
    # standard matrix-pencil projection: T = U^H H1 V Sigma^{-1}
    # (np.linalg.svd returns Vt = V^H, so V = Vt^H)
    T = (Ur.conj().T @ H1 @ Vtr.conj().T) / Sr[None, :]
    z = np.linalg.eigvals(T)
    on_circle = np.abs(np.abs(z) - 1.0) < circle_tol
    # The analytic filter is ONE-SIDED: every level appears exactly
    # once as e^{i D theta_j} (mirror suppressed to stop-band level),
    # so all on-circle angles in (-pi, pi] are kept as they are; no
    # conjugate folding, no halving.
    alpha = np.sort(np.angle(z[on_circle]))
    return alpha, {"hankel_blocks_used": int(H), "rank": rank,
                   "rank_auto": rank_auto, "n_on_circle": int(on_circle.sum())}


def _decode_decimated_angles(alphas_dec, D, theta_lo, theta_hi):
    """Invert theta -> D*theta mod 2 pi for the one-sided signal.

    The only ambiguity is the additive branch (alpha + 2 pi m)/D; every
    candidate inside [theta_lo, theta_hi] is returned. Because the
    voting streams use pairwise coprime D, two streams can only agree
    on the TRUE branch (Chinese remainder theorem), so aliasing ghosts
    are eliminated deterministically by the 2-of-3 vote.
    """
    D = int(D)
    two_pi = 2.0 * np.pi
    out = []
    for a in np.atleast_1d(alphas_dec):
        m_lo = int(np.floor((D * theta_lo - a) / two_pi)) - 1
        m_hi = int(np.ceil((D * theta_hi - a) / two_pi)) + 1
        for m in range(m_lo, m_hi + 1):
            theta = (two_pi * m + a) / D
            if theta_lo <= theta <= theta_hi:
                out.append(theta)
    return np.sort(np.asarray(out, dtype=float))




def bandpass_block_moments_exact(G, c_glob, e_glob, mu_low, mu_high,
                                 transition_frac=0.10, degree_frac=0.35):
    """Bandpass filter applied EXACTLY in cosine-coefficient space.

    The grid-based filter (evaluate / multiply / transform back with
    finite L) leaves a structured truncation residual: G_filt is then
    no longer an exact finite sum of cosines, and the rank-revealing
    pencil models the strongest coherent residual components as
    deterministic phantom poles offset ~half a level spacing from
    strong true poles. Building the window as an explicit degree-J
    cosine polynomial W_J and multiplying the two cosine series by
    exact coefficient convolution keeps the filtered sequence an EXACT
    sum of cosines with unchanged pole angles theta_j and amplitudes
    scaled by W_J(theta_j) >= 0. Phantoms are impossible by
    construction; residual stop-band leakage consists of exact poles
    at their true angles, which the Weyl rank cap and the coprime
    decode voting already handle.

    Conventions: g(theta) = sum_{n=0}^{L-1} G_n cos(n theta) and
    W_J(theta) = a_0/2 + sum_{m=1}^{J} a_m cos(m theta).

    Returns (G_filt, (theta_a, theta_b), n_usable): the first n_usable
    moments of G_filt are exact coefficients of W_J * g.
    """
    L = G.shape[0]
    x_low = np.clip((mu_low - c_glob) / e_glob, -1.0, 1.0)
    x_high = np.clip((mu_high - c_glob) / e_glob, -1.0, 1.0)
    theta_a = float(np.arccos(x_high))
    theta_b = float(np.arccos(x_low))
    span = theta_b - theta_a
    taper = transition_frac * span

    # window degree: enough to resolve the taper, bounded by the budget
    J = int(min(max(64, np.ceil(6.0 * np.pi / max(taper, 1e-9))),
                np.floor(degree_frac * (L - 1))))
    J = max(J, 1)

    # cosine coefficients of the tapered indicator (midpoint quadrature)
    n_grid = int(2 ** np.ceil(np.log2(max(16 * J, 8192))))
    tg = (np.arange(n_grid) + 0.5) * np.pi / n_grid
    W = np.zeros(n_grid)
    core = (tg >= theta_a + taper) & (tg <= theta_b - taper)
    W[core] = 1.0
    lo_t = (tg >= theta_a) & (tg < theta_a + taper)
    W[lo_t] = 0.5 * (1 - np.cos(np.pi * (tg[lo_t] - theta_a) / taper))
    hi_t = (tg > theta_b - taper) & (tg <= theta_b)
    W[hi_t] = 0.5 * (1 - np.cos(np.pi * (theta_b - tg[hi_t]) / taper))
    m = np.arange(J + 1)
    from scipy.fft import dct
    a = dct(W, type=2, norm=None)[: J + 1] / n_grid  # (2/N) sum W cos(m theta_q)
    # Lanczos sigma factors: suppress Gibbs ringing of the truncated
    # window polynomial (keeps W_J near-nonnegative, low stop band)
    a *= np.sinc(m / (J + 1))

    # impulse response of the sequence filter: W(theta) is the
    # frequency response over the "frequencies" theta_j carried by the
    # moment sequence G_n = sum_j A_j cos(n theta_j); filtering is a
    # symmetric convolution over n with h_m = h_{-m}, h_0 = a_0/2,
    # h_m = a_m/2 (m >= 1), applied to the even extension of G.
    c = np.concatenate([a[J:0:-1], a[:1], a[1:]]) * 0.5
    n_usable = L - 1 - J
    if n_usable < 8:
        raise ValueError("moment budget too small for window degree J")

    K = G.shape[1]
    G_filt = np.zeros((n_usable, K, K))
    ctr = (L - 1) + J  # index of n = 0 in the full convolution
    for i in range(K):
        for j in range(i, K):
            g = G[:, i, j]
            d = np.concatenate([g[L - 1:0:-1], g])   # even extension
            e = np.convolve(d, c)          # full, length 2L - 1 + 2J
            b = e[ctr:ctr + n_usable].copy()
            G_filt[:, i, j] = b
            if j != i:
                G_filt[:, j, i] = b
    return G_filt, (theta_a, theta_b), n_usable




def bandpass_block_moments_analytic(G, c_glob, e_glob, mu_low, mu_high,
                                    transition_frac=0.10, degree_frac=0.35):
    """One-sided (analytic) bandpass in coefficient space.

    The real moment sequence G_n = sum_j w_j cos(n theta_j) carries
    every level twice, at +/- theta_j. A real bandpass keeps both
    copies, and after decimation the +/- decode branches of DIFFERENT
    leakage poles can coincide across coprime streams on the lattice
    pi (m D2 + m' D1)/(D1 D2), producing deterministic ghosts. A
    ONE-SIDED window H(theta) ~ W(theta) 1_{theta > 0} removes the
    mirror line entirely: the filtered sequence is an exact sum of
    complex exponentials w_j W(theta_j) e^{i n theta_j} (mirror
    residual ~ stop-band level), each level appears ONCE, decode has a
    single additive branch theta = (alpha + 2 pi m)/D, and by
    coprimality no alias can agree across two streams.

    Returns (Z, (theta_a, theta_b), n_usable) with Z complex.
    """
    L = G.shape[0]
    x_low = np.clip((mu_low - c_glob) / e_glob, -1.0, 1.0)
    x_high = np.clip((mu_high - c_glob) / e_glob, -1.0, 1.0)
    theta_a = float(np.arccos(x_high))
    theta_b = float(np.arccos(x_low))
    span = theta_b - theta_a
    taper = transition_frac * span

    J = int(min(max(64, np.ceil(6.0 * np.pi / max(taper, 1e-9))),
                np.floor(degree_frac * (L - 1))))
    J = max(J, 1)

    n_grid = int(2 ** np.ceil(np.log2(max(16 * J, 8192))))
    # full circle grid for a genuinely one-sided response
    tg = -np.pi + (np.arange(n_grid) + 0.5) * (2 * np.pi / n_grid)
    W = np.zeros(n_grid)
    core = (tg >= theta_a + taper) & (tg <= theta_b - taper)
    W[core] = 1.0
    lo_t = (tg >= theta_a) & (tg < theta_a + taper)
    W[lo_t] = 0.5 * (1 - np.cos(np.pi * (tg[lo_t] - theta_a) / taper))
    hi_t = (tg > theta_b - taper) & (tg <= theta_b)
    W[hi_t] = 0.5 * (1 - np.cos(np.pi * (theta_b - tg[hi_t]) / taper))
    m = np.arange(-J, J + 1)
    # convolution gain for a pole e^{i n theta} is sum_m h_m e^{-i m theta},
    # so choose h_m = (1/2pi) int H(theta) e^{+i m theta} d theta.
    # Computed via FFT: with theta_q = -pi + (q + 1/2) 2 pi / n_grid,
    # h_m = e^{i m theta_0} * ifft(W)[m mod n_grid].
    theta0 = -np.pi + np.pi / n_grid
    hw = np.fft.ifft(W)
    h = np.exp(1j * m * theta0) * hw[np.mod(m, n_grid)]
    h *= np.sinc(m / (J + 1))  # Lanczos damping of the truncation

    n_usable = L - 1 - J
    if n_usable < 8:
        raise ValueError("moment budget too small for window degree J")

    K = G.shape[1]
    Z = np.zeros((n_usable, K, K), dtype=complex)
    ctr = (L - 1) + J
    for i in range(K):
        for j in range(i, K):
            g = G[:, i, j]
            d = np.concatenate([g[L - 1:0:-1], g])   # even extension
            e = np.convolve(d, h)
            b = e[ctr:ctr + n_usable].copy()
            Z[:, i, j] = b
            if j != i:
                Z[:, j, i] = b
    return Z, (theta_a, theta_b), n_usable


def tile_extract_decimated(
    G_filt,
    theta_a,
    theta_b,
    s_theta_min,
    n_probes,
    decimation_safety=0.6,
    hankel_dim_max=6000,
    rank_tol=1e-10,
    max_rank=500,
    circle_tol=0.08,
    cluster_frac=0.25,
    n_expected_window=None,
    rank_margin=1.15,
):
    """Multiplicity-preserving extraction of one bandpassed tile using
    TWO decimation factors with cross-validation.

    * Poles are decoded from the decimated pencil back into the padded
      filter window [theta_a, theta_b].
    * True poles decode identically for any decimation factor; aliasing
      ghosts (from decode ambiguity or filter leakage) land at
      D-dependent positions, so intersecting two runs with different D
      removes them deterministically.
    * Multiplicity: a (near-)degenerate level excited by K >= m probes
      contributes a rank-m block => m pencil eigenvalues within noise of
      the same angle; cluster size (capped at n_probes) after decoding,
      cross-validated as the minimum over the two decimations.

    cluster_frac * s_theta_min is the clustering/matching tolerance --
    far below the local mean level spacing s_theta_min.
    """
    # Weyl-informed rank cap: the filtered signal contains ~2x the
    # expected level count of the filter window as genuine pole
    # directions (conjugate pairs). Anything beyond that in the SVD is
    # suppressed stop-band signal and noise, whose on-circle roots
    # decode into the window as ghosts and drag cluster centers.
    if n_expected_window is not None:
        max_rank = int(min(max_rank,
                           np.ceil(rank_margin * n_expected_window) + 12))

    span = max(theta_b - theta_a, 1e-12)
    D1 = max(1, int(np.floor(decimation_safety * np.pi / span)))
    # Decimation factors must be PAIRWISE COPRIME: reflection ghosts of
    # a pole theta0 lie at 2*pi*m/D - theta0, so two streams D_i, D_j
    # share the ghost lattice 2*pi*t/gcd(D_i, D_j) - theta0. With
    # gcd = 1 the only shared position is the true pole itself, and
    # 2-of-3 voting rejects every ghost deterministically.
    from math import gcd
    D_list = [D1]
    for f in (0.8, 0.62):
        d = max(1, int(round(f * D1)))
        while d > 1 and any(gcd(d, x) != 1 for x in D_list):
            d -= 1
        if d not in D_list:
            D_list.append(d)
    tol = cluster_frac * s_theta_min

    obs_theta, obs_count, obs_stream, infos = [], [], [], []
    for s_idx, D in enumerate(D_list):
        alphas_dec, info = _decimated_pencil_angles(
            G_filt, D, hankel_dim_max=hankel_dim_max, rank_tol=rank_tol,
            max_rank=max_rank, circle_tol=circle_tol,
        )
        thetas = _decode_decimated_angles(alphas_dec, D, theta_a, theta_b)
        centers, counts = _cluster_sorted(thetas, tol)
        obs_theta.extend(centers.tolist())
        obs_count.extend(counts.tolist())
        obs_stream.extend([s_idx] * len(centers))
        info["decimation"] = int(D)
        info["n_decoded"] = int(len(centers))
        infos.append(info)

    n_streams = len(D_list)
    need = 1 if n_streams == 1 else 2
    order = np.argsort(obs_theta)
    obs_theta = np.asarray(obs_theta)[order]
    obs_count = np.asarray(obs_count)[order]
    obs_stream = np.asarray(obs_stream)[order]

    centers, counts = [], []
    i = 0
    while i < len(obs_theta):
        j = i + 1
        while j < len(obs_theta) and obs_theta[j] - obs_theta[j - 1] <= tol:
            j += 1
        streams = obs_stream[i:j]
        # True levels decode to the SAME position for every decimation
        # factor; aliasing ghosts are D-dependent. Requiring agreement
        # of >= 2 of the 3 streams rejects ghosts deterministically
        # while tolerating the loss of a level in any single stream.
        if len(np.unique(streams)) >= need:
            per_stream = [obs_count[i:j][streams == s].sum()
                          for s in np.unique(streams)]
            roots = float(np.median(per_stream))
            centers.append(float(np.median(obs_theta[i:j])))
            # one-sided signal: root count IS the multiplicity
            counts.append(int(min(max(1, round(roots)), n_probes)))
        i = j
    centers = np.asarray(centers)
    counts = np.asarray(counts, dtype=int)

    return centers, counts, {"decimations": infos,
                             "match_tol_theta": float(tol),
                             "n_final": int(len(centers))}


def band_resolvent_block_fdm(
    Kmat,
    Mmat,
    tile_edges,
    tile_lo,
    tile_hi,
    is_last_global_tile,
    sigma=None,
    n_moments=None,
    n_probes=13,
    hankel_blocks=110,
    transition_frac=0.10,
    pad_frac=0.75,
    seed=42,
    rank_tol=1e-10,
    max_rank=500,
    circle_tol=0.08,
    angle_cluster_frac=0.12,
    volume=None,
    surface=None,
    mean_curvature_integral=0.0,
    sigma_floor=5.0,
    resolution_factor=0.6,
    l_max=34000,
    verbose=True,
    checkpoint_path=None,
    checkpoint_every=2000,
):
    """One (band, seed) work unit of the gap-free tile-parallel scheme.

    Computes ONE moment sequence for the band [tile_edges[tile_lo],
    tile_edges[tile_hi]] with a band-optimal shift, then extracts every
    tile tile_lo..tile_hi-1 by bandpass + multiplicity-preserving
    pencil. Each tile keeps only its half-open core [t_k, t_{k+1})
    (the last GLOBAL tile keeps the closed interval), so results from
    different bands/tiles concatenate without duplicates or gaps.

    Returns (DataFrame[lambda, multiplicity, tile], info).
    """
    t0 = time.perf_counter()
    tile_edges = np.asarray(tile_edges, dtype=float)
    band_a = float(tile_edges[tile_lo])
    band_b = float(tile_edges[tile_hi])
    band_width = band_b - band_a

    pad0 = pad_frac * (tile_edges[tile_lo + 1] - tile_edges[tile_lo])
    band_a_pad = max(0.0, band_a - pad0)
    band_b_pad = band_b + pad_frac * (tile_edges[tile_hi] - tile_edges[tile_hi - 1])

    if sigma is None:
        sigma = optimal_band_sigma(band_a_pad, band_b_pad, sigma_floor=sigma_floor)
    if n_moments is None:
        if volume is None or surface is None:
            raise ValueError("auto n_moments needs volume and surface")
        n_moments = auto_moments_for_band(
            band_a_pad, band_b_pad, sigma, volume, surface,
            mean_curvature_integral, resolution_factor=resolution_factor,
            l_max=l_max,
        )
    if verbose:
        print(f"band lambda=[{band_a:.4f}, {band_b:.4f}], "
              f"sigma={sigma:.4f}, n_moments={n_moments}", flush=True)

    G, c_glob, e_glob = resolvent_chebyshev_block_moments(
        Kmat, Mmat, n_moments=n_moments, sigma=sigma,
        n_probes=n_probes, seed=seed, verbose=verbose,
        mu_headroom=1.25,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
    )

    rows, tile_info = [], []
    for k in range(tile_lo, tile_hi):
        lam_a = float(tile_edges[k])
        lam_b = float(tile_edges[k + 1])
        w = lam_b - lam_a
        # Padding may extend below lambda = 0: with mu_headroom > 1 the
        # corresponding mu is still interior to the Chebyshev interval,
        # which keeps the filter edge away from the interval endpoint.
        lam_a_pad = max(-0.15 * sigma, lam_a - pad_frac * w)
        lam_b_pad = lam_b + pad_frac * w

        mu_low = 1.0 / (lam_b_pad + sigma)
        mu_high = 1.0 / (lam_a_pad + sigma)
        Gf, (theta_a, theta_b), _n_usable = bandpass_block_moments_analytic(
            G, c_glob, e_glob, mu_low, mu_high, transition_frac=transition_frac,
        )

        # smallest mean level spacing inside the padded tile, in theta
        s_theta_min = np.inf
        if volume is not None and surface is not None:
            for lam in np.linspace(max(lam_a_pad, 1e-6), lam_b_pad, 9):
                mu = 1.0 / (lam + sigma)
                x = (mu - c_glob) / e_glob
                sin_t = np.sqrt(max(1.0 - min(x * x, 1.0), 1e-12))
                dlam_dtheta = e_glob * sin_t / mu**2
                rho = weyl_density(lam, volume, surface, mean_curvature_integral)
                s_theta_min = min(s_theta_min, 1.0 / (max(rho, 1e-12) * dlam_dtheta))
        else:
            s_theta_min = np.pi / n_moments

        n_exp_window = None
        if volume is not None and surface is not None:
            n_exp_window = float(
                weyl_counting_function(lam_b_pad, volume, surface, mean_curvature_integral)
                - weyl_counting_function(max(lam_a_pad, 0.0), volume, surface,
                                         mean_curvature_integral))
        alphas, mults, info = tile_extract_decimated(
            Gf, theta_a, theta_b, s_theta_min, n_probes,
            hankel_dim_max=hankel_blocks * n_probes,
            rank_tol=rank_tol, max_rank=max_rank, circle_tol=circle_tol,
            cluster_frac=angle_cluster_frac,
            n_expected_window=n_exp_window,
        )
        mu_vals = c_glob + e_glob * np.cos(alphas)
        good = np.isfinite(mu_vals) & (mu_vals > 0)
        lam_vals = 1.0 / mu_vals[good] - sigma
        mults = mults[good]

        last = is_last_global_tile and (k == tile_hi - 1)
        core = (lam_vals >= lam_a) & ((lam_vals <= lam_b) if last else (lam_vals < lam_b))
        lam_vals, m_vals = lam_vals[core], mults[core]

        for lv, mv in zip(lam_vals, m_vals):
            rows.append({"lambda": float(lv), "multiplicity": int(mv),
                         "tile": int(k), "tile_a": lam_a, "tile_b": lam_b})
        info.update({"tile": k, "tile_core": (lam_a, lam_b),
                     "n_core": int(len(lam_vals)),
                     "n_core_with_mult": int(m_vals.sum())})
        tile_info.append(info)
        if verbose:
            exp = ""
            if volume is not None and surface is not None:
                n_exp = (weyl_counting_function(lam_b, volume, surface, mean_curvature_integral)
                         - weyl_counting_function(lam_a, volume, surface, mean_curvature_integral))
                exp = f", expected~{n_exp:.1f}"
            print(f"  tile {k}: [{lam_a:.4f}, {lam_b:.4f}) "
                  f"kept={len(lam_vals)} (with mult: {int(m_vals.sum())}{exp})",
                  flush=True)

    df = pd.DataFrame(rows, columns=["lambda", "multiplicity", "tile", "tile_a", "tile_b"])
    df = df.sort_values("lambda").reset_index(drop=True)
    return df, {
        "algorithm": "gapfree_tile_parallel_resolvent_block_hankel",
        "wall_seconds": time.perf_counter() - t0,
        "sigma": float(sigma),
        "n_moments": int(n_moments),
        "band_lambda": (band_a, band_b),
        "band_lambda_padded": (band_a_pad, band_b_pad),
        "tile_lo": int(tile_lo),
        "tile_hi": int(tile_hi),
        "n_probes": int(n_probes),
        "seed": int(seed),
        "n_candidates": int(len(df)),
        "n_candidates_with_mult": int(df["multiplicity"].sum()) if len(df) else 0,
        "tiles": tile_info,
    }
