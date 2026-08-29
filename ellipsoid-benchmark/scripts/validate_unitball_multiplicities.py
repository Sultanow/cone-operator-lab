#!/usr/bin/env python3
"""Validate a computed unit-ball spectrum against the exact Dirichlet
eigenvalues of the unit ball, including their degeneracies.

For the unit ball the Dirichlet eigenvalues are exactly

    lambda_{l,n} = j_{l,n}^2 ,

where j_{l,n} is the n-th positive zero of the spherical Bessel
function j_l, and each level carries multiplicity 2l+1 (independent of
n). This is the sharpest available test of a multiplicity-resolving
eigensolver: the triaxial ellipsoid is generically simple, whereas the
ball reaches degeneracies of 2l+1 ~ 55 in the range resolved here.

The comparison is done on the CONTINUUM eigenvalues, so a systematic
FEM discretization shift is expected and is reported separately from
the structural verdict (counts, degeneracies, pairing). Use
--rel-tol to set the pairing window; it must be wide enough to absorb
the discretization shift (which grows with lambda) but well below the
local level spacing.

Usage:
  python validate_unitball_multiplicities.py \
      --spectrum .../full_spectrum_eigs_certified.txt \
      [--clusters .../full_spectrum_accepted.csv] \
      [--lambda-top 1018.0667] [--rel-tol 5e-3]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import spherical_jn


def spherical_bessel_zeros(l, lam_top, n_grid_per_unit=4.0):
    """All positive zeros j_{l,n} with j_{l,n}^2 <= lam_top."""
    x_top = np.sqrt(lam_top)
    # j_l has no zeros below ~l; start slightly above and scan
    x_lo = max(1e-6, l * 0.9)
    if x_lo >= x_top:
        return np.array([])
    n = max(int((x_top - x_lo) * n_grid_per_unit) + 10, 50)
    xs = np.linspace(x_lo, x_top, n)
    fs = spherical_jn(l, xs)
    zeros = []
    for i in range(len(xs) - 1):
        if fs[i] == 0.0:
            zeros.append(xs[i])
        elif fs[i] * fs[i + 1] < 0:
            zeros.append(brentq(lambda x: spherical_jn(l, x),
                                xs[i], xs[i + 1], xtol=1e-14, rtol=1e-15))
    return np.array(zeros)


def exact_unitball_levels(lam_top):
    """DataFrame of exact levels: lambda, l, n, multiplicity 2l+1."""
    rows = []
    l = 0
    while True:
        z = spherical_bessel_zeros(l, lam_top)
        z = z[z ** 2 <= lam_top]
        if len(z) == 0:
            break
        for n, zz in enumerate(np.sort(z), start=1):
            rows.append({"lambda": zz ** 2, "l": l, "n": n,
                         "multiplicity": 2 * l + 1})
        l += 1
    df = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spectrum", required=True,
                   help="one eigenvalue per line, multiplicities expanded")
    p.add_argument("--clusters", default=None,
                   help="optional full_spectrum_accepted.csv for a "
                        "per-cluster multiplicity comparison")
    p.add_argument("--lambda-top", type=float, default=None)
    p.add_argument("--rel-tol", type=float, default=5e-3)
    p.add_argument("--out", default="unitball_multiplicity_report.csv")
    args = p.parse_args()

    obs = np.sort(np.loadtxt(args.spectrum))
    lam_top = args.lambda_top if args.lambda_top else float(obs.max())
    obs = obs[obs <= lam_top * (1 + 1e-12)]

    exact = exact_unitball_levels(lam_top)
    n_exact = int(exact["multiplicity"].sum())
    print(f"exact levels (distinct): {len(exact)}   "
          f"with multiplicity: {n_exact}")
    print(f"observed levels:         {len(obs)}")
    print(f"count difference:        {len(obs) - n_exact:+d}")
    print(f"max degeneracy in range: {int(exact['multiplicity'].max())} "
          f"(l_max = {int(exact['l'].max())})")

    # group observed levels into clusters of near-equal values, then
    # pair clusters with exact levels
    if args.clusters:
        cl = pd.read_csv(args.clusters)
        lam_col = next(c for c in cl.columns if c.lower() in
                       ("lambda", "lam", "eigenvalue"))
        mult_col = next(c for c in cl.columns if "mult" in c.lower())
        obs_cl = cl[[lam_col, mult_col]].to_numpy()
        obs_cl = obs_cl[obs_cl[:, 0] <= lam_top * (1 + 1e-12)]
        obs_pos, obs_mult = obs_cl[:, 0], obs_cl[:, 1].astype(int)
    else:
        pos, mult = [], []
        i = 0
        while i < len(obs):
            j = i + 1
            while j < len(obs) and abs(obs[j] - obs[i]) <= 1e-9 * max(obs[i], 1):
                j += 1
            pos.append(float(np.median(obs[i:j])))
            mult.append(j - i)
            i = j
        obs_pos, obs_mult = np.array(pos), np.array(mult)
    order = np.argsort(obs_pos)
    obs_pos, obs_mult = obs_pos[order], obs_mult[order]
    print(f"observed distinct clusters: {len(obs_pos)}")

    rows, used = [], np.zeros(len(obs_pos), bool)
    for _, e in exact.iterrows():
        tol = args.rel_tol * max(e["lambda"], 1.0)
        cand = np.where(~used & (np.abs(obs_pos - e["lambda"]) <= tol))[0]
        if len(cand) == 0:
            rows.append({"lambda_exact": e["lambda"], "l": e["l"], "n": e["n"],
                         "mult_exact": e["multiplicity"], "lambda_obs": np.nan,
                         "mult_obs": 0, "status": "MISSING"})
            continue
        k = cand[np.argmin(np.abs(obs_pos[cand] - e["lambda"]))]
        used[k] = True
        status = ("ok" if obs_mult[k] == e["multiplicity"]
                  else ("under" if obs_mult[k] < e["multiplicity"] else "over"))
        rows.append({"lambda_exact": e["lambda"], "l": e["l"], "n": e["n"],
                     "mult_exact": e["multiplicity"],
                     "lambda_obs": obs_pos[k], "mult_obs": int(obs_mult[k]),
                     "status": status})
    rep = pd.DataFrame(rows)
    rep["rel_shift"] = (rep["lambda_obs"] - rep["lambda_exact"]) \
        / rep["lambda_exact"]

    ok = rep[rep.status == "ok"]
    print("\n== multiplicity verdict ==")
    print(rep["status"].value_counts().to_string())
    unmatched = int((~used).sum())
    print(f"observed clusters with no exact partner: {unmatched}")

    print("\n== degeneracy breakdown (exact mult -> recovered correctly) ==")
    g = rep.groupby("mult_exact").apply(
        lambda d: pd.Series({"levels": len(d),
                             "correct": int((d.status == "ok").sum())}))
    print(g.to_string())

    fin = rep.dropna(subset=["lambda_obs"])
    if len(fin):
        print("\n== discretization shift (observed vs continuum) ==")
        print(f"median rel. shift: {fin['rel_shift'].median():+.3e}")
        print(f"at lambda < 100:   "
              f"{fin[fin.lambda_exact < 100]['rel_shift'].median():+.3e}")
        print(f"at lambda > 900:   "
              f"{fin[fin.lambda_exact > 900]['rel_shift'].median():+.3e}")

    rep.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    verdict = (rep.status == "ok").all() and unmatched == 0
    print(f"STRUCTURAL VERDICT: {'PASS' if verdict else 'CHECK'}")


if __name__ == "__main__":
    main()
