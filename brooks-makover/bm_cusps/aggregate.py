#!/usr/bin/env python3
"""Collect results/*.json into a CSV and print the summary relevant for the
cusp/compactification comparison (lambda_1(S^bar) - lambda_1(S) vs. shortest cusp).
If a (n, seed) pair was run at two mesh sizes, Richardson-extrapolated (O(h^2)) values
are added."""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

files = [f for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/*.json", recursive=True))
         if ".invalid." not in f.rsplit("/", 1)[-1]
         and not f.endswith(".quarantine")
         and "/quarantine/" not in f.replace("\\", "/")]
rows = [json.load(open(f)) for f in files]
if not rows:
    sys.exit("no result files")
from version import SCHEMA_VERSION
old = [f for f, r in zip(files, rows) if r.get("schema_version", 0) < SCHEMA_VERSION]
if old:
    sys.exit("%d result file(s) have an old schema (< %d); regenerate with the current run_bm.py, e.g. %s"
             % (len(old), SCHEMA_VERSION, old[0]))
eligible = [(f, r) for f, r in zip(files, rows) if r.get("quality", {}).get("analysis_eligible") is True]
dropped = [f for f, r in zip(files, rows) if r.get("quality", {}).get("analysis_eligible") is not True]
if dropped:
    print("quality filter: excluding %d non-converged/non-eligible result(s), e.g. %s" % (len(dropped), dropped[0]))
if not eligible:
    sys.exit("all matching results are non-converged or not analysis-eligible")
files = [x[0] for x in eligible]
rows = [x[1] for x in eligible]

cols = ["n", "seed", "h", "V", "genus", "min_cusp_length", "max_cusp_length",
        "lambda1_thick", "lambda1_neumann", "lambda1_cusped", "lambda1_compact",
        "delta_compact_minus_cusped", "delta_compact_minus_thick",
        "liou_area_err", "weyl_slope", "weyl_slope_expected", "u_min_at_height1", "eps_h_central",
        "girth", "n_short_cycles", "n_tangle_walks", "n_cusps_short", "systole",
        "n_geodesics_below_3", "mu2", "gap", "nb_rho2", "ramanujan_adj", "wallclock_s"]
GRAPH_COLS = ["girth", "n_short_cycles", "n_tangle_walks", "n_cusps_short", "systole",
              "n_geodesics_below_3", "mu2", "gap", "nb_rho2", "ramanujan_adj"]


def flat(r):
    d = {k: r.get(k) for k in cols if k in r}
    d["liou_area_err"] = abs(r["liouville"]["area_compact"] - r["liouville"]["area_compact_exact"]) / r["liouville"]["area_compact_exact"]
    d["weyl_slope"] = r["weyl"]["slope"]
    d["weyl_slope_expected"] = r["weyl"].get("slope_expected")
    d["u_min_at_height1"] = min(p["u_height1_mean"] for p in r["liouville"]["u_per_cusp"])
    d["eps_h_central"] = r["liouville"].get("eps_h_central")
    for c in GRAPH_COLS:
        d[c] = r.get("graph", {}).get(c)
    return d


flat_rows_all = [flat(r) for r in rows]

# Statistical identity is (n, seed), not a mesh width.  Keep all resolutions only
# for convergence/Richardson diagnostics; ensemble summaries use exactly one record
# per random surface, namely the finest available mesh.
by_surface = defaultdict(list)
for d in flat_rows_all:
    by_surface[(int(d["n"]), int(d["seed"]))].append(d)
flat_rows = []
duplicate_groups = []
for key, rr in sorted(by_surface.items()):
    hmin = min(float(x.get("h", np.inf)) for x in rr)
    finest = [x for x in rr if float(x.get("h", np.inf)) == hmin]
    if len(finest) != 1:
        sys.exit("duplicate result files at the same finest resolution for surface %s, h=%g" % (key, hmin))
    flat_rows.append(finest[0])
    if len(rr) > 1:
        duplicate_groups.append((key, sorted(float(x["h"]) for x in rr), hmin))

# Canonical ensemble CSV: one row per independent surface.  Preserve all mesh
# resolutions separately for convergence studies.
for name, data in (("summary.csv", flat_rows), ("summary_all_resolutions.csv", flat_rows_all)):
    with open(name, "w") as f:
        f.write(",".join(cols) + "\n")
        for d in data:
            f.write(",".join("" if d.get(c) is None else str(d[c]) for c in cols) + "\n")
print("wrote summary.csv (%d independent surfaces) and summary_all_resolutions.csv (%d files)"
      % (len(flat_rows), len(flat_rows_all)))
if duplicate_groups:
    print("resolution selection for ensemble statistics: finest h per (n, seed)")
    for (n, seed), hs, hmin in duplicate_groups:
        print("  surface n=%d seed=%d: h=%s -> selected h=%g" % (n, seed, hs, hmin))
print()

# ---- Richardson extrapolation over mesh sizes (same n, seed) ------------------------
by = defaultdict(dict)
for d in flat_rows_all:
    by[(d["n"], d["seed"])][d["h"]] = d
rich = []
for key, hs in by.items():
    if len(hs) >= 2:
        h1, h2 = sorted(hs)[:2]
        a, b = hs[h1], hs[h2]
        ex = {}
        for q in ("lambda1_compact", "lambda1_cusped", "lambda1_thick"):
            if a.get(q) is not None and b.get(q) is not None:
                ex[q] = (a[q] * h2 ** 2 - b[q] * h1 ** 2) / (h2 ** 2 - h1 ** 2)
        rich.append((key, h1, h2, ex))
if rich:
    print("Richardson-extrapolated (h->0) values:")
    for (n, s), h1, h2, ex in rich:
        print("  n=%4d seed=%3d (h=%g,%g): " % (n, s, h1, h2) + "  ".join("%s=%.6f" % kv for kv in ex.items()))
    print()

# ---- summary per n ---------------------------------------------------------------------
print("%5s %4s | %6s %6s | %9s %9s %9s | %8s %8s | %7s %7s" % (
    "n", "#", "V", "genus", "lam1_thk", "lam1_S", "lam1_Sbar", "P(<1/4)", "d(Sbar-S)", "min u", "t[s]"))
for n in sorted({d["n"] for d in flat_rows}):
    g = [d for d in flat_rows if d["n"] == n]
    S = [d["lambda1_cusped"] for d in g if d["lambda1_cusped"] is not None]
    dl = [d["delta_compact_minus_cusped"] for d in g if d["delta_compact_minus_cusped"] is not None]
    print("%5d %4d | %6.2f %6.1f | %9.4f %9.4f %9.4f | %8.2f %8.4f | %7.2f %7.0f" % (
        n, len(g), np.mean([d["V"] for d in g]), np.mean([d["genus"] for d in g]),
        np.mean([d["lambda1_thick"] for d in g]), np.mean(S) if S else float("nan"),
        np.mean([d["lambda1_compact"] for d in g]), len(S) / len(g),
        np.mean(dl) if dl else float("nan"), np.mean([d["u_min_at_height1"] for d in g]),
        np.mean([d["wallclock_s"] for d in g])))

# ---- dependence on the shortest cusp -------------------------------------------------
print("\nlambda_1(S^bar) - lambda_1(S_thick) grouped by shortest cusp length k_min:")
for k in sorted({d["min_cusp_length"] for d in flat_rows}):
    g = [d for d in flat_rows if d["min_cusp_length"] == k]
    print("  k_min=%3d  #=%3d  mean delta=%.4f  mean min u(height-1)=%.3f" % (
        k, len(g), np.mean([d["delta_compact_minus_thick"] for d in g]), np.mean([d["u_min_at_height1"] for d in g])))

# ---- graph -> surface transfer: correlations with lambda_1(S^bar) --------------------
print("\nPearson correlation of lambda_1(S^bar) with graph features (per n, needs >= 8 samples):")
for n in sorted({d["n"] for d in flat_rows}):
    g = [d for d in flat_rows if d["n"] == n and d.get("mu2") is not None]
    if len(g) < 8:
        continue
    y = np.array([d["lambda1_compact"] for d in g])
    out = []
    for c in ("gap", "systole", "n_short_cycles", "n_tangle_walks", "n_cusps_short", "girth"):
        x = np.array([float(d[c]) for d in g])
        if x.std() > 0:
            out.append("%s:%+.2f" % (c, np.corrcoef(x, y)[0, 1]))
    print("  n=%4d (#%d): " % (n, len(g)) + "  ".join(out))
