#!/usr/bin/env python3
"""
analyze.py -- statistical evaluation of the FEM runs (results/*.json).

  python analyze.py h1 [--glob 'results/*.json'] [--boot 2000]
      Finite-size law of lambda_1(S^bar):  m(n) = L + C n^{-alpha}  and  m(n) = L + C / log n
      with FREE limit L, fitted to the per-n medians; cluster bootstrap (resample surfaces
      within each n) gives CIs for L, C, alpha; then the test H0: L = 1/4 is reported as a
      z-score.  1/4 is never built into the fit.

  python analyze.py h4 [--glob 'results/*.json'] [--boot 2000] [--plot h4_u_vs_k.png]
      Cusp-level point cloud (k_c, u_c(1)): conditional mean / variance per k, coloured
      by n; residual regression  u = F(k) + beta . (girth, gap, n_tangle_walks, n_cusps_short)
      with F(k) nonparametric (per-k means); cluster bootstrap BY SURFACE for the betas.
      Locality reading: beta ~ 0  <=>  global graph geometry adds nothing once k is known.

Every bootstrap here resamples whole surfaces, never individual cusps.
"""
import argparse
import glob
import json
import sys
from collections import defaultdict

import numpy as np


def load(pattern):
    files = sorted(glob.glob(pattern))
    rows = [json.load(open(f)) for f in files]
    if not rows:
        sys.exit("no result files for %s" % pattern)
    check_schema(rows, files)
    return rows

def check_schema(rows, files):
    from version import SCHEMA_VERSION
    old = [f for f, r in zip(files, rows) if r.get("schema_version", 0) < SCHEMA_VERSION]
    if old:
        sys.exit("%d result file(s) have an old schema (< %d) and must be regenerated with the current "
                 "run_bm.py, e.g. %s" % (len(old), SCHEMA_VERSION, old[0]))



# ----------------------------------------------------------------------------
# H1
# ----------------------------------------------------------------------------
def _fit_power(ns, m):
    from scipy.optimize import curve_fit
    f = lambda n, L, C, a: L + C * n ** (-a)
    p, _ = curve_fit(f, ns, m, p0=[0.25, 1.0, 0.5], maxfev=20000)
    return p


def _fit_log(ns, m):
    X = np.column_stack([np.ones_like(ns, dtype=float), 1.0 / np.log(ns)])
    L, C = np.linalg.lstsq(X, m, rcond=None)[0]
    return np.array([L, C])


def h1(args):
    rows = load(args.glob)
    by_n = defaultdict(list)
    for r in rows:
        by_n[r["n"]].append(r["lambda1_compact"])
    ns = np.array(sorted(by_n))
    if len(ns) < 3:
        sys.exit("H1 needs at least 3 distinct n (have %s)" % ns.tolist())
    med = np.array([np.median(by_n[n]) for n in ns])
    print("per n: " + "  ".join("n=%d (#%d) median=%.5f" % (n, len(by_n[n]), m) for n, m in zip(ns, med)))
    rng = np.random.default_rng(0)
    boots = {"power": [], "log": []}
    for _ in range(args.boot):
        mb = np.array([np.median(rng.choice(by_n[n], size=len(by_n[n]), replace=True)) for n in ns])
        try:
            boots["power"].append(_fit_power(ns, mb))
        except RuntimeError:
            pass
        boots["log"].append(_fit_log(ns, mb))
    for name, fit in (("L + C n^-alpha", _fit_power), ("L + C / log n", _fit_log)):
        p = fit(ns, med)
        B = np.array(boots["power" if "alpha" in name else "log"])
        sd = B.std(axis=0)
        resid = med - (p[0] + p[1] * ns ** (-p[2]) if len(p) == 3 else p[0] + p[1] / np.log(ns))
        print("\nmodel %s:" % name)
        print("  L     = %.5f +- %.5f   (H0: L = 1/4  ->  z = %+.2f)" % (p[0], sd[0], (p[0] - 0.25) / sd[0] if sd[0] > 0 else float("nan")))
        print("  C     = %.4f +- %.4f" % (p[1], sd[1]))
        if len(p) == 3:
            print("  alpha = %.4f +- %.4f" % (p[2], sd[2]))
        print("  RMS residual of medians = %.2e   (bootstrap fits: %d)" % (np.sqrt(np.mean(resid ** 2)), len(B)))
    print("\nP(lambda_1(S^bar) > 1/4) per n: " + "  ".join("%d:%.2f" % (n, np.mean(np.array(by_n[n]) > 0.25)) for n in ns))


# ----------------------------------------------------------------------------
# H4
# ----------------------------------------------------------------------------
FEATS = ("girth", "gap", "n_tangle_walks", "n_cusps_short")


def cusp_table(rows):
    """One record per cusp: n, surface id, k, mean u and sup|u| on the height-1 horocycle, graph features."""
    T = []
    for sid, r in enumerate(rows):
        g = r.get("graph", {})
        feats = [float(g.get(f) if g.get(f) is not None else np.nan) for f in FEATS]
        for p in r["liouville"]["u_per_cusp"]:
            T.append([r["n"], sid, p["k"], p["u_height1_mean"], p["u_height1_sup"]] + feats)
    return np.array(T, dtype=float)


def _kbin(k):
    """k itself for k <= 8, dyadic bins [8,16), [16,32), ... above (singletons carry no
    information after within-bin centring)."""
    k = np.asarray(k, dtype=int)
    return np.where(k <= 8, k, 8 * 2 ** np.floor(np.log2(np.maximum(k, 8) / 8)).astype(int))


def _regress(T):
    """u = F(k) + beta . feats  with F(k) = per-k-bin mean (absorbed by centring within bins)."""
    k = _kbin(T[:, 2])
    u = T[:, 3].copy()
    X = T[:, 5:].copy()
    ok = ~np.isnan(X).any(axis=1)
    k, u, X = k[ok], u[ok], X[ok]
    for kk in np.unique(k):                       # within-k centring removes F(k) exactly
        m = k == kk
        u[m] -= u[m].mean()
        X[m] -= X[m].mean(axis=0)
    beta = np.linalg.lstsq(X, u, rcond=None)[0]
    r2 = 1 - np.sum((u - X @ beta) ** 2) / max(np.sum(u ** 2), 1e-300)
    return beta, r2


def c0_table(rows):
    """Per cusp: the density c0 of g_bar at the filled puncture in the cusp's canonical coordinate
    w = exp(2 pi i z / k), inferred from u on the horocycle of (actual) length ell via
        u(ell) = 1/2 log c0 + log(2 pi / ell) - 2 pi / ell + O(exp(-4 pi / ell)),
    i.e. g_bar ~ c0 |dw|^2 near w = 0 (Poincare unit disc: c0 = 4).  Uses the row with the
    smallest ell (>= 1) where the correction is negligible; also returns the prediction error
    at the other stored ell from the exact disc formula with conformal radius rho = 2/sqrt(c0):
        u(ell) = log( (4 pi / ell) r0 rho / (rho^2 - r0^2) ),  r0 = exp(-2 pi / ell)."""
    out = []
    for sid, r in enumerate(rows):
        for p in r["liouville"]["u_per_cusp"]:
            ual = p.get("u_at_length", {})
            if not ual:
                continue
            ell1, (u1, _, la1) = min(((float(e), v) for e, v in ual.items()), key=lambda t: t[0])
            half_log_c0 = u1 - np.log(2 * np.pi / la1) + 2 * np.pi / la1
            c0 = float(np.exp(2 * half_log_c0))
            rho = 2 / np.sqrt(c0)
            errs = {}
            for e, (u, _, la) in ual.items():
                r0 = np.exp(-2 * np.pi / la)
                pred = np.log((4 * np.pi / la) * r0 * rho / (rho ** 2 - r0 ** 2)) if rho > r0 else np.nan
                errs[e] = u - pred
            out.append(dict(n=r["n"], sid=sid, k=p["k"], c0=c0, rho=float(rho), pred_err=errs))
    return out


def h4(args):
    rows = load(args.glob)
    T = cusp_table(rows)
    C = c0_table(rows)
    if C:
        c0 = np.array([c["c0"] for c in C]); kk = np.array([c["k"] for c in C]); nn_ = np.array([c["n"] for c in C])
        print("\nconformal density c0 at the filled punctures (unit-disc value 4):  mean %.3f  sd %.3f  [%d cusps]"
              % (c0.mean(), c0.std(), len(c0)))
        for kb in np.unique(_kbin(kk)):
            m = _kbin(kk) == kb
            print("  %-14s c0 = %.3f +- %.3f   (per n: %s)" % ("k=%d" % kb if kb <= 8 else "k in [%d,%d)" % (kb, 2 * kb),
                  c0[m].mean(), c0[m].std(), "  ".join("%d:%.3f" % (n, c0[m & (nn_ == n)].mean()) for n in np.unique(nn_) if (m & (nn_ == n)).any())))
        for e in ("2", "4", "8", "16"):
            errs = np.array([c["pred_err"][e] for c in C if e in c["pred_err"] and np.isfinite(c["pred_err"][e])])
            if len(errs):
                print("  disc-formula prediction error at ell=%s: mean %+.4f  sd %.4f  [%d cusps]" % (e, errs.mean(), errs.std(), len(errs)))
    ns = np.unique(T[:, 0]).astype(int)
    print("%d surfaces, %d cusp observations, n in %s" % (len(rows), len(T), ns.tolist()))
    print("\nconditional moments of u_c(1) given k (all n pooled; per-n means in brackets):")
    kb = _kbin(T[:, 2])
    ks = np.unique(kb)
    for kk in ks:
        m = kb == kk
        per_n = "  ".join("n=%d:%.3f" % (n, T[m & (T[:, 0] == n), 3].mean()) for n in ns if (m & (T[:, 0] == n)).any())
        lab = "k=%3d" % kk if kk <= 8 else "k in [%d,%d)" % (kk, 2 * kk)
        print("  %-14s #=%4d  mu=%+.4f  sigma=%.4f   [%s]" % (lab, m.sum(), T[m, 3].mean(), T[m, 3].std(), per_n))
    # residual regression + cluster bootstrap by surface
    beta, r2 = _regress(T)
    rng = np.random.default_rng(0)
    sids = np.unique(T[:, 1]).astype(int)
    B = []
    for _ in range(args.boot):
        pick = rng.choice(sids, size=len(sids), replace=True)
        Tb = np.vstack([T[T[:, 1] == s] for s in pick])
        try:
            B.append(_regress(Tb)[0])
        except np.linalg.LinAlgError:
            pass
    B = np.array(B)
    print("\nresidual regression  u_c(1) - F(k) = beta . (%s)   [cluster bootstrap by surface, %d resamples]"
          % (", ".join(FEATS), len(B)))
    for i, f in enumerate(FEATS):
        lo, hi = np.percentile(B[:, i], [2.5, 97.5]) if len(B) else (np.nan, np.nan)
        flag = "" if lo <= 0 <= hi else "  <-- differs from 0"
        print("  beta[%-14s] = %+.4f   95%% CI [%+.4f, %+.4f]%s" % (f, beta[i], lo, hi, flag))
    print("  R^2 of the residual regression = %.4f   (locality <=> ~0)" % r2)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for n in ns:
            m = T[:, 0] == n
            ax.scatter(T[m, 2], T[m, 3], s=14, alpha=0.6, label="n=%d" % n)
        mu = [T[kb == kk, 3].mean() for kk in ks]
        ax.plot([kk if kk <= 8 else kk * np.sqrt(2) for kk in ks], mu, "k-", lw=1, label="E[u | k-bin]")
        ax.set_xscale("log"); ax.set_xlabel("cusp length k_c"); ax.set_ylabel("u_c (conformal factor) on the height-1 horocycle")
        ax.set_title("Compactification distortion vs cusp length (H4)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(args.plot, dpi=150)
        print("plot -> %s" % args.plot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["h1", "h4"])
    ap.add_argument("--glob", default="results/*.json")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()
    (h1 if args.what == "h1" else h4)(args)


if __name__ == "__main__":
    main()
