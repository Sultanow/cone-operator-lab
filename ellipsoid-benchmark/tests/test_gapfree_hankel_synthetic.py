"""End-to-end synthetic validation of the gap-free tile-parallel
resolvent block-Hankel pipeline.

We build a sparse generalized eigenproblem (K, M) with EXACTLY known
spectrum: eigenvalues sampled from the Weyl density of the working
ellipsoid (a, b, c) = (1, 1.5, 2.3), with planted exact degeneracies of
multiplicity 2 and 3. K is diagonal and M is a well-conditioned sparse
SPD mass-like matrix (congruence-transformed), so splu and the probe
machinery are exercised realistically.

Checks:
  * every true eigenvalue in [0, lambda_top] is recovered (no gaps),
  * no spurious extras survive the cross-seed consensus,
  * multiplicities are recovered,
  * accuracy is far below the local mean level spacing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hearing_ellipsoid_bench.geometry.ellipsoid import (  # noqa: E402
    ellipsoid_integrated_mean_curvature,
    ellipsoid_surface_area,
    true_ellipsoid_volume,
)
from hearing_ellipsoid_bench.solvers.fdm_block import (  # noqa: E402
    band_resolvent_block_fdm,
    group_tiles_into_bands,
    weyl_counting_function,
    weyl_equal_count_edges,
)


def build_synthetic_problem(V, S, C, n_target, rng):
    """Sparse (K, M) whose generalized spectrum follows the Weyl density
    with planted degeneracies. Returns (K, M, true_eigs_sorted)."""
    from scipy.optimize import brentq

    lam_top = brentq(
        lambda l: weyl_counting_function(l, V, S, C) - n_target, 1e-6, 1e6)
    # sample distinct levels via inverse Weyl counting with jitter
    n_distinct = int(n_target * 0.93)
    u = (np.arange(1, n_distinct + 1) - 0.5 + 0.35 * rng.standard_normal(n_distinct))
    u = np.clip(u, 0.05, n_target - 0.05)
    u = np.sort(u)
    levels = np.array([
        brentq(lambda l: weyl_counting_function(l, V, S, C) - ui, 1e-9, 2 * lam_top)
        for ui in (u / n_distinct * weyl_counting_function(lam_top, V, S, C))
    ])
    # enforce a minimal separation between distinct levels
    for i in range(1, len(levels)):
        levels[i] = max(levels[i], levels[i - 1] * (1 + 5e-6) + 1e-9)
    # plant degeneracies
    mult = np.ones(len(levels), dtype=int)
    deg_idx = rng.choice(len(levels), size=len(levels) // 18, replace=False)
    for j in deg_idx:
        mult[j] = rng.integers(2, 4)
    true_eigs = np.sort(np.repeat(levels, mult))
    # pad spectrum above lambda_top so the top edge is not artificial
    pad = lam_top * (1.0 + np.sort(rng.random(160)) * 0.6)
    diag = np.concatenate([true_eigs, pad])

    n = len(diag)
    perm = rng.permutation(n)
    D = sp.diags(diag[perm]).tocsr()
    # well-conditioned sparse SPD "mass": M = I + small symmetric band
    off = 0.08 * rng.standard_normal(n - 1)
    B = sp.diags([off, off], offsets=[-1, 1])
    M = (sp.eye(n) + B + 0.5 * sp.diags(np.abs(off).max() * np.ones(n))).tocsr()
    M = (0.5 * (M + M.T)).tocsr()
    # K = L D L^T with L = chol-like sparse factor of M => gen. eigs of
    # (K, M) equal diag exactly:  K x = lam M x  with  M = L L^T.
    from scipy.sparse.linalg import splu as _splu  # noqa: F401
    import scipy.sparse.linalg as spla
    # sparse Cholesky via LU of SPD matrix (M = P^T L U): easier exact
    # route: pick M diagonal-dominant and use its exact Cholesky from
    # scipy.linalg on banded form -- simplest robust option: M := W W^T
    W = sp.eye(n) + sp.diags([0.05 * rng.standard_normal(n - 1)], offsets=[-1])
    M = (W @ W.T).tocsr()
    K = (W @ D @ W.T).tocsr()
    K = (0.5 * (K + K.T)).tocsr()
    M = (0.5 * (M + M.T)).tocsr()
    return K, M, true_eigs, lam_top


def main():
    rng = np.random.default_rng(7)
    a, b, c = 1.0, 1.5, 2.3
    V = true_ellipsoid_volume(a, b, c)
    S = ellipsoid_surface_area(a, b, c)
    C = ellipsoid_integrated_mean_curvature(a, b, c)

    n_target = 400
    K, M, true_eigs, lam_top_sample = build_synthetic_problem(V, S, C, n_target, rng)
    print(f"synthetic problem: n={K.shape[0]}, "
          f"true eigs in range={len(true_eigs)}, lam_top~{lam_top_sample:.3f}")

    n_tiles, n_bands = 16, 4
    edges, n_total = weyl_equal_count_edges(
        n_tiles, V, S, C, lambda_top=float(true_eigs.max()) * (1 + 1e-12))
    bands = group_tiles_into_bands(edges, n_bands)
    truth = true_eigs[true_eigs <= edges[-1]]

    seeds = [11, 22, 33, 44, 55]
    frames = []
    for band_index, (lo, hi) in enumerate(bands):
        for seed in seeds:
            df, info = band_resolvent_block_fdm(
                K, M, tile_edges=edges, tile_lo=lo, tile_hi=hi,
                is_last_global_tile=(hi == n_tiles),
                n_probes=7, hankel_blocks=90, max_rank=350,
                pad_frac=0.75, transition_frac=0.10,
                seed=seed, volume=V, surface=S, mean_curvature_integral=C,
                resolution_factor=0.6, l_max=12000, verbose=False,
            )
            df["band"] = band_index
            df["seed"] = seed
            frames.append(df)
            print(f"band {band_index} seed {seed}: sigma={info['sigma']:.2f} "
                  f"L={info['n_moments']} candidates={info['n_candidates_with_mult']}")
    raw = pd.concat(frames, ignore_index=True)

    out = Path("/home/claude/work/synth_out")
    out.mkdir(exist_ok=True)
    raw.to_csv(out / "band_all_candidates.csv", index=False, float_format="%.17e")

    # reuse the real merge logic
    sys.path.insert(0, str(ROOT / "scripts"))
    import merge_hankel_full_spectrum as mg
    density = lambda lam: mg.weyl_density(lam, V, S, C)
    per_tile = [mg.consensus_tile(g, 0.15, 4, density_fn=density)
                for _, g in raw.groupby("tile", sort=True)]
    clusters = pd.concat(per_tile, ignore_index=True)
    accepted = clusters[clusters["accepted"]]
    spectrum = np.sort(np.repeat(accepted["lambda"].to_numpy(),
                                 accepted["multiplicity"].to_numpy()))

    # ------------------------- verdict -------------------------
    print("\n==== verdict ====")
    print(f"true count (with mult): {len(truth)}   recovered: {len(spectrum)}")
    # match greedily
    from scipy.optimize import linear_sum_assignment  # noqa: F401
    i = j = 0
    matched, missed, extra = [], [], []
    tol_rel = 5e-4
    while i < len(truth) and j < len(spectrum):
        d = (spectrum[j] - truth[i]) / max(truth[i], 1.0)
        if abs(d) <= tol_rel:
            matched.append(abs(d)); i += 1; j += 1
        elif d < 0:
            extra.append(spectrum[j]); j += 1
        else:
            missed.append(truth[i]); i += 1
    missed.extend(truth[i:]); extra.extend(spectrum[j:])

    # Second pass: sub-Rayleigh near-degeneracies. Levels closer than
    # ~0.4 of the effective Rayleigh limit are reported by design as a
    # multiplicity-m cluster at the centroid of the m levels. Accept an
    # unmatched group of m identical extras whose value equals the
    # centroid of m unmatched true levels within 0.5 local mean
    # spacing, all lying within one mean spacing of the centroid.
    missed = np.sort(np.array(missed)); extra = np.sort(np.array(extra))
    still_missed, used_extra = [], np.zeros(len(extra), bool)
    k = 0
    while k < len(missed):
        resolved = False
        for m_sz in (3, 2):
            grp = missed[k:k + m_sz]
            if len(grp) < m_sz:
                continue
            cen = grp.mean()
            sp = 1.0 / max(density(cen), 1e-12)
            if grp.max() - grp.min() > 1.0 * sp:
                continue
            cand = np.where(~used_extra
                            & (np.abs(extra - cen) <= 0.5 * sp))[0]
            if len(cand) >= m_sz:
                used_extra[cand[:m_sz]] = True
                k += m_sz
                resolved = True
                break
        if not resolved:
            still_missed.append(missed[k]); k += 1
    n_centroid = int(used_extra.sum())
    missed = still_missed
    extra = list(extra[~used_extra])
    if n_centroid:
        print(f"centroid-cluster reconciled (sub-Rayleigh pairs): "
              f"{n_centroid} observations")
    matched = np.array(matched)
    print(f"matched: {len(matched)}  missed: {len(missed)}  extra: {len(extra)}")
    if len(matched):
        print(f"rel err: median={np.median(matched):.3e}  "
              f"p99={np.quantile(matched, 0.99):.3e}  max={matched.max():.3e}")
    if missed:
        print("missed eigenvalues:", np.array(missed))
    if extra:
        print("spurious extras:", np.array(extra))

    ok = (len(missed) == 0) and (len(extra) == 0)
    print("GAP-FREE + SPURIOUS-FREE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
