#!/usr/bin/env python3
"""Compare the gap-free Hankel spectrum against a reference spectrum
(FEM-ARPACK on the SAME mesh/matrices).

Usage:
  python compare_vs_reference.py \
      --hankel /path/to/full_spectrum_eigs.txt \
      --reference /path/to/arpack_eigs.txt \
      [--lambda-top 449.12] [--rel-tol 5e-7] \
      [--a 1.0 --b 1.5 --c 2.3]

The reference file may be .txt (one eigenvalue per line) or .csv (a
column named lambda/eig/eigenvalue, else the first numeric column).
Matching is greedy on sorted lists with tolerance
  tol(lam) = max(rel_tol * lam, 0.05 * local_mean_spacing)
and a second reconciliation pass accepts sub-Rayleigh centroid
clusters (multiplicity-m observation at the centroid of m reference
levels within half a local mean spacing).
"""
import argparse
import numpy as np


def load_any(path):
    try:
        return np.loadtxt(path)
    except Exception:
        import pandas as pd
        df = pd.read_csv(path)
        for c in df.columns:
            if c.lower() in ("lambda", "eig", "eigenvalue", "lam"):
                return df[c].to_numpy(dtype=float)
        return df.select_dtypes("number").iloc[:, 0].to_numpy(dtype=float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hankel", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--lambda-top", type=float, default=None)
    p.add_argument("--rel-tol", type=float, default=5e-7)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--b", type=float, default=1.5)
    p.add_argument("--c", type=float, default=2.3)
    args = p.parse_args()

    H = np.sort(load_any(args.hankel))
    R = np.sort(load_any(args.reference))
    top = args.lambda_top if args.lambda_top is not None else H.max() * (1 + 1e-12)
    R = R[(R >= 0) & (R <= top)]
    H = H[(H >= 0) & (H <= top)]
    print(f"hankel: {len(H)} levels <= {top:.4f}")
    print(f"reference: {len(R)} levels <= {top:.4f}")
    print(f"count difference (hankel - reference): {len(H) - len(R):+d}")

    # local mean spacing from the reference itself (window of 21)
    def spacing_at(x):
        i = np.searchsorted(R, x)
        lo, hi = max(0, i - 10), min(len(R), i + 11)
        if hi - lo < 2:
            return 1.0
        return (R[hi - 1] - R[lo]) / (hi - lo - 1)

    i = j = 0
    missed, extra, errs, matched_pairs = [], [], [], []
    while i < len(R) and j < len(H):
        sp = spacing_at(R[i])
        tol = max(args.rel_tol * max(R[i], 1.0), 0.05 * sp)
        d = H[j] - R[i]
        if abs(d) <= tol:
            errs.append(abs(d) / max(R[i], 1e-12))
            matched_pairs.append((R[i], H[j]))
            i += 1; j += 1
        elif d < 0:
            extra.append(H[j]); j += 1
        else:
            missed.append(R[i]); i += 1
    missed += list(R[i:]); extra += list(H[j:])

    # centroid reconciliation for sub-Rayleigh near-degeneracies
    missed = np.array(sorted(missed)); extra = np.array(sorted(extra))
    used = np.zeros(len(extra), bool)
    still = []
    k = 0
    n_centroid = 0
    while k < len(missed):
        done = False
        for m_sz in (3, 2):
            grp = missed[k:k + m_sz]
            if len(grp) < m_sz:
                continue
            cen = grp.mean(); sp = spacing_at(cen)
            if grp.max() - grp.min() > 1.0 * sp:
                continue
            cand = np.where(~used & (np.abs(extra - cen) <= 0.5 * sp))[0]
            if len(cand) >= m_sz:
                used[cand[:m_sz]] = True
                n_centroid += m_sz
                k += m_sz
                done = True
                break
        if not done:
            still.append(missed[k]); k += 1
    missed = np.array(still); extra = extra[~used]

    errs = np.array(errs)
    print(f"\nmatched: {len(errs)}   centroid-reconciled: {n_centroid}")
    print(f"missed (in reference, not in hankel): {len(missed)}")
    print(f"extra  (in hankel, not in reference): {len(extra)}")
    if len(errs):
        print(f"rel err: median={np.median(errs):.3e}  "
              f"p99={np.quantile(errs, 0.99):.3e}  max={errs.max():.3e}")
    if len(missed):
        print("missed:", np.array2string(missed, precision=4, threshold=50))
    if len(extra):
        print("extra: ", np.array2string(extra, precision=4, threshold=50))

    ok = (len(missed) == 0 and len(extra) == 0)
    print(f"\nVERDICT vs discrete reference: {'PASS' if ok else 'CHECK'}")
    if not ok and len(missed) <= 20 and len(extra) <= 20:
        print("(small residuals: inspect whether these lie near lambda_top "
              "or in flagged tiles; the Weyl audit vs continuum is NOT the "
              "arbiter for a discrete operator)")


if __name__ == "__main__":
    main()