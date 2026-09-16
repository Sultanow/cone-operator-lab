#!/usr/bin/env python3
"""
graph_scan.py -- cheap combinatorial screening of Brooks-Makover samples (no FEM).

Stage 1 of the funnel  10^6 graphs -> 10^3 candidate surfaces -> FEM spectra:
for every seed the ribbon graph is generated exactly as run_bm.py does (same rng),
its graph features are written to a CSV, and optionally the seeds with the most
extreme values of a feature are exported as a params file for slurm_bm.sbatch.

Writes two files: <out>.csv (flat, for ranking / pandas) and <out>.jsonl (the complete
graph_features() dictionary per seed -- the lossless raw record that run_bm.py reuses).

Usage:
  python graph_scan.py --n 128 --seeds 0 100000 --W 10 --out scan_n128.csv
  python graph_scan.py --n 128 --seeds 0 100000 --out scan_n128.csv \\
         --select nb_rho2 --top 50 --bottom 50 --h 0.1 --params params_extremes.txt
"""
import argparse
import csv
import json
import sys
import time

import numpy as np

from bmgraph import graph_features
from version import __version__
from bmsurf import BMSurface

COLS = ["n", "seed", "V", "genus", "k_max_fraction", "n_cusps_len1", "n_cusps_len2", "n_cusps_short",
        "girth", "n_short_cycles", "n_tangle_walks", "c1", "c2", "c3", "c4", "c5", "c6",
        "systole", "n_geodesics_below_2", "n_geodesics_below_3", "n_geodesics_below_cut",
        "mu2", "gap", "mu_min", "ramanujan_adj", "nb_rho2", "ramanujan_nb"]


def row_of(f, seed):
    r = {c: f.get(c) for c in COLS}
    r["seed"] = seed
    for l in range(1, 7):
        r["c%d" % l] = f["simple_cycles"].get(l, 0)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 1000], help="seed range [a, b)")
    ap.add_argument("--W", type=int, default=10, help="max word / cycle length enumerated")
    ap.add_argument("--no-spectra", action="store_true", help="skip adjacency / non-backtracking spectra")
    ap.add_argument("--out", required=True)
    ap.add_argument("--select", default=None, help="feature to rank by (e.g. nb_rho2, systole, n_tangle_walks)")
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--bottom", type=int, default=0)
    ap.add_argument("--h", type=float, default=0.1)
    ap.add_argument("--L0", type=float, default=1.0)
    ap.add_argument("--params", default="params_extremes.txt")
    a = ap.parse_args()

    print("bm_cusps %s" % __version__, file=sys.stderr)
    rows, t0 = [], time.time()
    jsonl_path = a.out[:-4] + ".jsonl" if a.out.endswith(".csv") else a.out + ".jsonl"
    with open(a.out, "w", newline="") as fh, open(jsonl_path, "w") as fj:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for i, seed in enumerate(range(a.seeds[0], a.seeds[1])):
            surf = BMSurface.random(a.n, np.random.default_rng(seed))
            if surf.g < 2:
                continue
            f = graph_features(surf, W=a.W, spectra=not a.no_spectra)
            f["seed"] = seed
            fj.write(json.dumps(f) + "\n")           # lossless raw record (schema = graph_features)
            r = row_of(f, seed)
            rows.append(r)
            w.writerow(r)
            if (i + 1) % 100 == 0:
                print("%d samples, %.1f s" % (i + 1, time.time() - t0), file=sys.stderr, flush=True)
    print("wrote %s (%d rows, %.1f s) and %s" % (a.out, len(rows), time.time() - t0, jsonl_path))

    if a.select:
        vals = np.array([r[a.select] if r[a.select] is not None else np.nan for r in rows], dtype=float)
        order = np.argsort(vals)
        pick = list(order[:a.bottom]) + list(order[::-1][:a.top])
        with open(a.params, "w") as fh:
            for i in pick:
                fh.write("%d %d %g %g\n" % (a.n, rows[i]["seed"], a.h, a.L0))
        print("selected %d extreme seeds by %s -> %s  (%s in [%.4g, %.4g])"
              % (len(pick), a.select, a.params, a.select, np.nanmin(vals), np.nanmax(vals)))


if __name__ == "__main__":
    main()
