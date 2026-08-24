#!/usr/bin/env python3
"""Adjudicate position disputes between the certified Hankel spectrum
and a reference spectrum via single inertia probes.

A "dispute" is a pair (h, r) of one Hankel level and one reference
level with |h - r| below half the local mean spacing, where neither
matches the other list within tolerance. For each pair, one LDL^T
inertia evaluation at the midpoint m = (h + r)/2 decides the side on
which the true discrete eigenvalue lies: if N(m) - N(lo) equals the
number of certified levels up to and including the disputed one, the
eigenvalue is below m (favoring the smaller candidate), else above.

Usage:
  python adjudicate_positions.py \
    --hankel .../full_spectrum_eigs_certified.txt \
    --reference .../arnoldi_..._eigs.txt \
    --data-root /home/esul01/data \
    --a 1.0 --b 1.5 --c 2.3 --mesh-h 0.06 --order 2 \
    [--rel-tol 5e-7] [--out adjudication_report.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from certify_inertia import InertiaOracle  # same directory
from hearing_ellipsoid_bench.fem.assembly import load_or_create_problem


def load_any(path):
    try:
        return np.loadtxt(path)
    except Exception:
        df = pd.read_csv(path)
        for c in df.columns:
            if c.lower() in ("lambda", "eig", "eigenvalue", "lam"):
                return df[c].to_numpy(dtype=float)
        return df.select_dtypes("number").iloc[:, 0].to_numpy(dtype=float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hankel", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--b", type=float, default=1.5)
    p.add_argument("--c", type=float, default=2.3)
    p.add_argument("--mesh-h", type=float, default=0.06)
    p.add_argument("--order", type=int, default=2)
    p.add_argument("--rel-tol", type=float, default=5e-7)
    p.add_argument("--out", default="adjudication_report.csv")
    args = p.parse_args()

    H = np.sort(load_any(args.hankel))
    R = np.sort(load_any(args.reference))
    top = H.max() * (1 + 1e-12)
    R = R[(R > 0) & (R <= top)]

    def spacing_at(x, arr):
        i = np.searchsorted(arr, x)
        lo, hi = max(0, i - 10), min(len(arr), i + 11)
        return (arr[hi - 1] - arr[lo]) / max(hi - lo - 1, 1)

    # unmatched-in-both within tolerance -> then pair leftovers
    used_h = np.zeros(len(H), bool)
    miss = []
    for r in R:
        tol = max(args.rel_tol * r, 0.0)
        i = np.searchsorted(H, r)
        best, bd = -1, np.inf
        for j in (i - 1, i):
            if 0 <= j < len(H) and not used_h[j]:
                d = abs(H[j] - r)
                if d < bd:
                    best, bd = j, d
        if best >= 0 and bd <= tol:
            used_h[best] = True
        else:
            miss.append(r)
    extra = H[~used_h]

    pairs = []
    extra_left = list(extra)
    for r in miss:
        sp = spacing_at(r, R)
        cand = [e for e in extra_left if abs(e - r) < 0.5 * sp]
        if cand:
            e = min(cand, key=lambda x: abs(x - r))
            extra_left.remove(e)
            pairs.append((float(e), float(r)))
    print(f"disputed pairs: {len(pairs)}   "
          f"(unpaired ref-missing: {len(miss) - len(pairs)}, "
          f"unpaired hankel-extra: {len(extra_left)})")
    if not pairs:
        print("nothing to adjudicate.")
        return

    print("loading FEM problem ...", flush=True)
    _, K, M, _ = load_or_create_problem(
        args.data_root, a=args.a, b=args.b, c=args.c,
        mesh_size=args.mesh_h, order=args.order, force_remesh=False)
    oracle = InertiaOracle(K, M)

    rows = []
    for h, r in pairs:
        lo, hi = (h, r) if h < r else (r, h)
        mid = 0.5 * (lo + hi)
        # number of certified levels strictly below mid, from the
        # certified list itself:
        n_expect_if_low = int(np.searchsorted(H, mid))
        n_below = oracle.count_below(mid)
        # if the true eigenvalue lies below mid, N(mid) equals the
        # certified count below mid computed with the disputed level on
        # the LOW side; the certified list H already places it at h.
        if h < r:
            verdict = "hankel" if n_below == n_expect_if_low else "reference"
        else:
            verdict = "reference" if n_below == n_expect_if_low else "hankel"
        rows.append({"hankel": h, "reference": r, "midpoint": mid,
                     "inertia_below_mid": n_below,
                     "certified_below_mid_if_hankel": n_expect_if_low,
                     "winner": verdict})
        print(f"  h={h:.6f} vs r={r:.6f}  -> winner: {verdict}",
              flush=True)
    oracle.close()

    rep = pd.DataFrame(rows)
    rep.to_csv(args.out, index=False)
    print(f"\nwinners: {rep['winner'].value_counts().to_dict()}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
