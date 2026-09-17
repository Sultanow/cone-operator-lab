"""Independent Fenchel--Nielsen gluing validation on a closed genus-2 surface.

Two identical pairs of pants with symmetric cuff length
    l_s = 2 arccosh(1 + sqrt(2))
are glued along all three cuffs.  A cuff with 2m boundary nodes is identified by
orientation reversal plus a discrete twist.  The production quantity called
``lambda_1`` below is always the first positive eigenvalue ``vals[1]``.

The scan is useful as a *numerical validation experiment*, not as a theorem about
which genus-2 surface globally maximizes lambda_1.  The Kravchuk--Mazac--Pal result
provides a very sharp genus-2 upper bound near the Bolza value; it does not by
itself prove exact global maximality of the Bolza surface.

When the mesh parameter m is changed, refinement must preserve the *normalized
Fenchel--Nielsen twist* t in [0,1), not the raw boundary-node shift j.  This module
therefore exposes twists as normalized fractions and converts them to the nearest
compatible node shift for each mesh.
"""
from __future__ import annotations

import argparse
import numpy as np
from scipy.sparse import block_diag, coo_matrix
from scipy.sparse.linalg import eigsh
try:
    from .pants import build_pants
except ImportError:  # direct script execution
    from pants import build_pants

BOLZA = 3.838887258
KMP_BOUND = 3.8388976481


def twist_step_from_fraction(t: float, K: int) -> int:
    """Nearest discrete cuff-node shift representing normalized twist t mod 1."""
    return int(round((float(t) % 1.0) * K)) % K


def realized_twist_fraction(t: float, K: int) -> float:
    return twist_step_from_fraction(t, K) / float(K)


def glue_genus2(l, twist_steps=None, *, twist_fractions=None, m=16, h=0.045, k=6):
    """Build a closed genus-2 FEM system and return area, spectrum, dof count.

    Exactly one of ``twist_steps`` or ``twist_fractions`` may be supplied.
    ``twist_fractions`` is preferred for refinement studies because it preserves
    the same geometric twist when ``m`` changes (up to the discrete boundary grid).
    """
    if twist_steps is not None and twist_fractions is not None:
        raise ValueError("specify twist_steps or twist_fractions, not both")
    if twist_steps is None and twist_fractions is None:
        twist_fractions = [0.0, 0.0, 0.0]

    P = build_pants(l, l, l, m=m, h=h)
    N = len(P['coords'])
    S = block_diag([P['S'], P['S']]).tocsr()
    M = block_diag([P['M'], P['M']]).tocsr()
    parent = list(range(2 * N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(3):
        chain = P['cuffs'][i][0]
        K = len(chain)  # = 2m
        if twist_fractions is not None:
            j0 = twist_step_from_fraction(twist_fractions[i], K)
        else:
            j0 = int(twist_steps[i]) % K
        for kk in range(K):
            union(chain[kk], N + chain[(j0 - kk) % K])

    roots = [find(x) for x in range(2 * N)]
    uniq = sorted(set(roots))
    remap = {r: i for i, r in enumerate(uniq)}
    dof = np.array([remap[roots[x]] for x in range(2 * N)])
    ndof = len(uniq)
    R = coo_matrix((np.ones(2 * N), (np.arange(2 * N), dof)), shape=(2 * N, ndof)).tocsr()
    Sd = (R.T @ S @ R).tocsr()
    Md = (R.T @ M @ R).tocsr()
    md = np.asarray(Md.diagonal()).ravel()
    good = np.where(md > 1e-12)[0]
    Sd = Sd[good][:, good]
    Md = Md[good][:, good]
    vals, _ = eigsh(Sd, k=k, M=Md, sigma=-1e-6, which='LM')
    return float(Md.sum()), np.sort(vals.real), ndof


def scan_symmetric_twists(m=16, h=0.045, step=2, k=6):
    """Scan equal twists on all three cuffs; return records using true lambda_1."""
    ls = 2 * np.arccosh(1 + np.sqrt(2))
    K = 2 * m
    recs = []
    for j in range(0, K, step):
        t = j / K
        area, vals, ndof = glue_genus2(ls, twist_fractions=[t, t, t], m=m, h=h, k=k)
        recs.append(dict(j=j, t=t, area=area, ndof=ndof,
                         lambda1=float(vals[1]),
                         first_cluster_mean=float(np.mean(vals[1:4])),
                         vals=vals))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=16)
    ap.add_argument('--h', type=float, default=0.045)
    ap.add_argument('--step', type=int, default=2)
    ap.add_argument('--refine-m', type=int, default=24)
    ap.add_argument('--refine-h', type=float, default=0.032)
    args = ap.parse_args()

    ls = 2 * np.arccosh(1 + np.sqrt(2))
    print(f"symmetric cuff length l_s = 2 arccosh(1+sqrt2) = {ls:.5f}\n")

    area, vals, ndof = glue_genus2(ls, twist_fractions=[0, 0, 0], m=args.m, h=args.h)
    print(f"[twist 0] dofs={ndof} area={area:.4f} (4pi={4*np.pi:.4f}) lambda_0={vals[0]:.2e}")
    print(f"          low spectrum: {np.array2string(vals, precision=4, floatmode='fixed')}\n")

    print("twist sweep (symmetric decomposition):")
    print(f"{'t':>8} | {'lambda_1':>12} | {'mean(lambda_1..3)':>20} | {'<= KMP bound?':>14}")
    print("-" * 64)
    recs = scan_symmetric_twists(m=args.m, h=args.h, step=args.step)
    for r in recs:
        ok = 'yes' if r['lambda1'] <= KMP_BOUND + 1e-3 else 'NO (!!)'
        print(f"{r['t']:8.3f} | {r['lambda1']:12.5f} | {r['first_cluster_mean']:20.5f} | {ok:>14}")
    best = max(recs, key=lambda r: r['lambda1'])
    print("-" * 64)
    print(f"\nmax of sampled lambda_1 values = {best['lambda1']:.5f} at t={best['t']:.6f}")
    print(f"Bolza reference = {BOLZA:.9f}; KMP near-sharp upper bound = {KMP_BOUND:.10f}")
    print("No exact global-maximality claim is inferred from this numerical sweep or from the bound.")

    # Preserve the normalized twist under mesh refinement.  Do NOT reuse raw j.
    tbest = best['t']
    Kfine = 2 * args.refine_m
    tfine = realized_twist_fraction(tbest, Kfine)
    _, vf, ndof_f = glue_genus2(ls, twist_fractions=[tbest] * 3,
                                m=args.refine_m, h=args.refine_h, k=6)
    print(f"\nfiner mesh at same normalized twist: requested t={tbest:.6f}, "
          f"realized t={tfine:.6f} (m={args.refine_m}, dofs={ndof_f})")
    print(f"  lambda_1={vf[1]:.6f}; first three positive eigenvalues="
          f"{np.array2string(vf[1:4], precision=6, floatmode='fixed')}")


if __name__ == '__main__':
    main()
