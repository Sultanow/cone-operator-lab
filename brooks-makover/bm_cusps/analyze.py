#!/usr/bin/env python3
"""
analyze.py -- statistical evaluation of the FEM runs (results/*.json).

  python analyze.py h1 [--glob 'results/*.json'] [--boot 2000]
      Finite-size law of lambda_1(S^bar):  m(n) = L + C n^{-alpha}  and  m(n) = L + C / log n
      with FREE limit L, fitted to the per-n medians; cluster bootstrap (resample surfaces
      within each n) gives CIs for L, C, alpha; then the test H0: L = 1/4 is reported as a
      z-score.  1/4 is never built into the fit.

  python analyze.py h4 [--glob 'results/*.json'] [--boot 2000] [--plot h4_u_vs_k.png]
      Cusp-level diagnostics plus the H4 locality diagnostic on puncture density c0:
      c0 = F(k) + A_n + beta . (girth, gap, n_tangle_walks, n_cusps_short),
      with nonparametric k-bin and n fixed effects.  The reported partial R^2 asks how much
      global graph covariates add after controlling for both cusp length and surface size.
      Bootstrap resampling is clustered BY SURFACE and stratified within n.  This is a
      numerical diagnostic, not a proof of locality.

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
    for f, r in zip(files, rows):
        r["_source_file"] = f
    return rows


def select_unique_surfaces(rows):
    """Return one statistically independent record per generated surface.

    A Brooks--Makover surface is determined here by (n, seed).  Multiple h values are
    discretisations of the same random object, not independent samples.  For H4 we use
    the finest available mesh (smallest h); controlled extrapolation of c0 is deliberately
    not attempted until its h-asymptotics have been validated separately.
    """
    by = defaultdict(list)
    for r in rows:
        by[(int(r["n"]), int(r["seed"]))].append(r)
    out = []
    duplicate_groups = []
    for key, rr in sorted(by.items()):
        hs = [float(x.get("h", np.inf)) for x in rr]
        hmin = min(hs)
        finest = [x for x in rr if float(x.get("h", np.inf)) == hmin]
        if len(finest) > 1:
            names = [x.get("_source_file", "?") for x in finest]
            raise ValueError("duplicate result files at the same finest resolution for surface %s, h=%g: %s"
                             % (key, hmin, names))
        out.append(finest[0])
        if len(rr) > 1:
            duplicate_groups.append((key, sorted(hs, reverse=True), hmin))
    if duplicate_groups:
        print("resolution selection: %d files -> %d independent surfaces; using finest h per (n, seed)"
              % (len(rows), len(out)))
        for key, hs, hmin in duplicate_groups:
            print("  surface n=%d seed=%d: h=%s -> selected h=%g" % (key[0], key[1], hs, hmin))
    return out

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
    rows = select_unique_surfaces(load(args.glob))
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
    """Per cusp: puncture density c0 plus the global graph covariates used by H4.

    c0 is inferred from the smallest stored fixed-length horocycle.  The surface id is
    assigned *after* resolution selection, so all cusps from one random surface form one
    bootstrap cluster.
    """
    out = []
    for sid, r in enumerate(rows):
        g = r.get("graph", {})
        feats = {f: float(g.get(f)) if g.get(f) is not None else np.nan for f in FEATS}
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
            out.append(dict(n=r["n"], seed=r["seed"], sid=sid, k=p["k"], c0=c0,
                            rho=float(rho), pred_err=errs, feats=feats))
    return out


def _regress_c0(C):
    """Fit c0 = F(k-bin) + A_n + beta . global_features.

    k-bin and n are treated as categorical fixed effects.  ``partial_r2`` measures the
    incremental explanatory power of the global graph covariates beyond those controls.
    The fitted n coefficients are descriptive finite-size effects relative to the smallest
    observed n, not evidence for an asymptotic law.
    """
    if not C:
        raise ValueError("no c0 observations available")
    k = _kbin(np.array([c["k"] for c in C], dtype=int))
    n = np.array([int(c["n"]) for c in C], dtype=int)
    y = np.array([c["c0"] for c in C], dtype=float)
    Xfeat = np.array([[c["feats"][f] for f in FEATS] for c in C], dtype=float)
    ok = np.isfinite(y) & ~np.isnan(Xfeat).any(axis=1)
    k, n, y, Xfeat = k[ok], n[ok], y[ok], Xfeat[ok]
    if not len(y):
        raise ValueError("no c0 observations with complete graph covariates")

    # Categorical controls: intercept + all non-baseline k bins + all non-baseline n.
    kvals = np.unique(k)
    nvals = np.unique(n)
    controls = [np.ones(len(y))]
    control_names = ["intercept"]
    for kk in kvals[1:]:
        controls.append((k == kk).astype(float)); control_names.append("k=%s" % kk)
    for nn in nvals[1:]:
        controls.append((n == nn).astype(float)); control_names.append("n=%s" % nn)
    Xctrl = np.column_stack(controls)
    Xfull = np.column_stack([Xctrl, Xfeat])

    pfull = np.linalg.lstsq(Xfull, y, rcond=None)[0]
    pctrl = np.linalg.lstsq(Xctrl, y, rcond=None)[0]
    resid_full = y - Xfull @ pfull
    resid_ctrl = y - Xctrl @ pctrl
    sse_full = float(resid_full @ resid_full)
    sse_ctrl = float(resid_ctrl @ resid_ctrl)
    # If the controls already explain y to machine precision, there is no residual
    # variance for graph features to explain; report zero rather than a numerically
    # meaningless large negative ratio of two round-off-level SSEs.
    scale = max(float(y @ y), 1.0)
    partial_r2 = 0.0 if sse_ctrl <= 1e-24 * scale else 1.0 - sse_full / sse_ctrl
    beta = pfull[-len(FEATS):]

    n_effects = {int(nvals[0]): 0.0}
    offset = 1 + max(len(kvals) - 1, 0)
    for j, nn in enumerate(nvals[1:]):
        n_effects[int(nn)] = float(pfull[offset + j])
    return beta, partial_r2, n_effects


def h4(args):
    rows = select_unique_surfaces(load(args.glob))
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
    # H4 diagnostic: control explicitly for both cusp length and surface size n.
    beta, partial_r2, n_effects = _regress_c0(C)
    rng = np.random.default_rng(0)
    sid_to_n = {int(c["sid"]): int(c["n"]) for c in C}
    sids_by_n = {nn: np.array(sorted(s for s, n0 in sid_to_n.items() if n0 == nn), dtype=int)
                 for nn in sorted(set(sid_to_n.values()))}
    B = []
    for _ in range(args.boot):
        Cb = []
        # Preserve the observed number of independent surfaces in every n-class.
        for nn, sids in sids_by_n.items():
            pick = rng.choice(sids, size=len(sids), replace=True)
            for s0 in pick:
                Cb.extend(c for c in C if int(c["sid"]) == int(s0))
        try:
            B.append(_regress_c0(Cb)[0])
        except (np.linalg.LinAlgError, ValueError):
            pass
    B = np.array(B)
    print("\nH4 diagnostic: c0 = F(k-bin) + A_n + beta . (%s)" % ", ".join(FEATS))
    print("  n fixed effects are included; bootstrap is clustered by surface and stratified within n (%d resamples)" % len(B))
    for i, f in enumerate(FEATS):
        lo, hi = np.percentile(B[:, i], [2.5, 97.5]) if len(B) else (np.nan, np.nan)
        flag = "" if lo <= 0 <= hi else "  <-- differs from 0"
        print("  beta[%-14s] = %+.4f   95%% CI [%+.4f, %+.4f]%s" % (f, beta[i], lo, hi, flag))
    print("  partial R^2 added by global graph covariates after controlling for k and n = %.4f" % partial_r2)
    if len(n_effects) > 1:
        base = min(n_effects)
        print("  descriptive n fixed effects relative to n=%d: %s" %
              (base, "  ".join("n=%d:%+.4f" % (nn, a) for nn, a in sorted(n_effects.items()) if nn != base)))
    print("  interpretation: beta near zero / small partial R^2 is evidence consistent with locality, not a proof.")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        cc0 = np.array([c["c0"] for c in C]); ck = np.array([c["k"] for c in C]); cn = np.array([c["n"] for c in C])
        ckb = _kbin(ck)
        cks = np.unique(ckb)
        for n in np.unique(cn):
            m = cn == n
            ax.scatter(ck[m], cc0[m], s=14, alpha=0.6, label="n=%d" % n)
        mu = [cc0[ckb == kk].mean() for kk in cks]
        ax.plot([kk if kk <= 8 else kk * np.sqrt(2) for kk in cks], mu, "k-", lw=1, label="E[c0 | k-bin]")
        ax.axhline(4.0, color="0.5", ls="--", lw=1, label="unit-disc c0=4")
        ax.set_xscale("log"); ax.set_xlabel("cusp length k_c"); ax.set_ylabel("puncture density c0")
        ax.set_title("Puncture density vs cusp length (H4 locality test)")
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
