#!/usr/bin/env python3
"""
selftest.py -- run this in the directory you intend to launch the campaign from.

It fails loudly if the directory holds an old code state (the review regression tests):
  * geodesic count (n=16, seed=4, W=10): 7 below ell_cut = 2 arccosh(11/2) = 4.779
  * closed non-backtracking walk counts agree with the Ihara trace formula (n=8, seed=2)
  * graph_scan.py writes CSV *and* JSONL
  * run_bm.py output carries the current schema (u_per_cusp, eps_h_*), analyze.py reads it
  * the disc formula reproduces u at ell=2 from ell=1 to < 1e-2 on a small surface

Usage:  python selftest.py            (about 30 s)
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from version import SCHEMA_VERSION, __version__          # noqa: E402
from bmsurf import BMSurface                              # noqa: E402
from bmgraph import cycle_and_length_features, nonbacktracking  # noqa: E402

fails = 0


def check(cond, msg):
    global fails
    print(("  ok   " if cond else "  FAIL ") + msg)
    fails += 0 if cond else 1


print("bm_cusps %s, schema %d, directory %s" % (__version__, SCHEMA_VERSION, HERE))

# 1) geodesic count regression (review point 1)
surf = BMSurface.random(16, np.random.default_rng(4))
f = cycle_and_length_features(surf, W=10)
check(abs(f["ell_cut"] - 2 * math.acosh(11 / 2)) < 1e-12, "completeness cut ell_cut = 2 arccosh((W+1)/2)")
check(f["n_geodesics_below_cut"] == 7, "n=16 seed=4: %d geodesics below the cut (expect 7)" % f["n_geodesics_below_cut"])
check(all(x < f["ell_cut"] for x in f["length_spectrum"]), "length_spectrum contains only lengths below the cut")

# 2) trace-formula cross-check of the walk enumeration
surf8 = BMSurface.random(8, np.random.default_rng(2))
f8 = cycle_and_length_features(surf8, W=8)
B = nonbacktracking(surf8).toarray()
tr = [np.trace(np.linalg.matrix_power(B, l)) for l in range(1, 9)]
mob = {1: 1, 2: -1, 3: -1, 4: 0, 5: -1, 6: 1, 7: -1, 8: 0}
ok = True
for l in range(1, 9):
    P = sum(mob[l // d] * tr[d - 1] for d in range(1, l + 1) if l % d == 0) // l
    ok &= (f8["simple_cycles"].get(l, 0) + f8["tangle_walks"].get(l, 0)) == P // 2
check(ok, "closed NB walk counts = Ihara/Moebius counts for l <= 8")

tmp = tempfile.mkdtemp()
# 3) scan writes CSV and JSONL
subprocess.run([sys.executable, "graph_scan.py", "--n", "8", "--seeds", "0", "3", "--out", os.path.join(tmp, "scan.csv")],
               check=True, capture_output=True)
check(os.path.exists(os.path.join(tmp, "scan.csv")) and os.path.exists(os.path.join(tmp, "scan.jsonl")),
      "graph_scan.py writes scan.csv and scan.jsonl")

# 4) run_bm.py schema + reuse of the scan record + analyze.py end to end
out = os.path.join(tmp, "r.json")
subprocess.run([sys.executable, "run_bm.py", "--n", "8", "--seed", "2", "--h", "0.2", "--neig", "6", "--quiet",
                "--graph-from", os.path.join(tmp, "scan.jsonl"), "--out", out], check=True, capture_output=True)
r = json.load(open(out))
check(r.get("schema_version") == SCHEMA_VERSION, "run_bm.py output has schema_version %s" % r.get("schema_version"))
check("u_per_cusp" in r["liouville"] and "eps_h_central" in r["liouville"], "liouville block uses u_per_cusp / eps_h_*")
check(r["graph"]["graph_source"].startswith("scan:"), "graph features reused from the scan JSONL")
check(r["dtn"].get("certified") is False, "cusp DtN result carries certified=false")
p = subprocess.run([sys.executable, "analyze.py", "h4", "--glob", out, "--boot", "5"], capture_output=True, text=True)
check(p.returncode == 0 and "conformal density" in p.stdout, "analyze.py h4 reads the result")
# 5) disc formula on this small surface
errs = []
for c in r["liouville"]["u_per_cusp"]:
    ual = c["u_at_length"]
    if "1" in ual and "2" in ual:
        u1, _, l1 = ual["1"]; u2, _, l2 = ual["2"]
        rho = 2 / math.sqrt(math.exp(2 * (u1 - math.log(2 * math.pi / l1) + 2 * math.pi / l1)))
        r0 = math.exp(-2 * math.pi / l2)
        errs.append(abs(u2 - math.log((4 * math.pi / l2) * r0 * rho / (rho ** 2 - r0 ** 2))))
check(bool(errs) and max(errs) < 1e-2, "disc formula predicts u(2) from u(1) within 1e-2 (max err %.1e)" % (max(errs) if errs else float("nan")))

print("\n%s" % ("ALL CHECKS PASSED -- this directory holds bm_cusps %s" % __version__ if fails == 0
                else "%d CHECK(S) FAILED -- do not launch the campaign from this directory" % fails))
sys.exit(1 if fails else 0)
