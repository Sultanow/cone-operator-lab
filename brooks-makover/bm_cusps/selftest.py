#!/usr/bin/env python3
"""
selftest.py -- run this in the directory you intend to launch the campaign from.

It fails loudly if the directory holds an old code state (the review regression tests):
  * geodesic count (n=16, seed=4, W=10): 7 below ell_cut = 2 arccosh(11/2) = 4.779
  * closed non-backtracking walk counts agree with the Ihara trace formula (n=8, seed=2)
  * graph_scan.py writes CSV *and* JSONL
  * run_bm.py output carries the current schema (u_per_cusp, eps_h_*), analyze.py reads it
  * the disc profile reproduces u at ell=2 from ell=1 to < 1e-2 on a small surface
  * aggregate.py deduplicates mesh resolutions by (n, seed)
  * SLURM result validation rejects mismatched/stale outputs
  * H4 n fixed effects remove a synthetic size-confounded signal

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

from version import GRAPH_FEATURE_VERSION, SCAN_SCHEMA_VERSION, SCHEMA_VERSION, __version__  # noqa: E402
from bmsurf import BMSurface                              # noqa: E402
from bmgraph import cycle_and_length_features, nonbacktracking  # noqa: E402
from analyze import FEATS, _regress_c0, select_unique_surfaces  # noqa: E402

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
first_scan = json.loads(open(os.path.join(tmp, "scan.jsonl")).readline())
check(first_scan.get("scan_schema_version") == SCAN_SCHEMA_VERSION and
      first_scan.get("graph_feature_version") == GRAPH_FEATURE_VERSION,
      "scan JSONL carries current scan/graph-feature provenance")

# 3b) stale scan records must be rejected, never wrapped in a current result schema
stale = os.path.join(tmp, "stale.jsonl")
with open(os.path.join(tmp, "scan.jsonl")) as fi, open(stale, "w") as fo:
    for line in fi:
        x = json.loads(line); x.pop("scan_schema_version", None); x.pop("graph_feature_version", None)
        fo.write(json.dumps(x) + "\n")
pstale = subprocess.run([sys.executable, "run_bm.py", "--n", "8", "--seed", "2", "--h", "0.2", "--neig", "6", "--quiet",
                         "--graph-from", stale, "--out", os.path.join(tmp, "must_not_exist.json")],
                        capture_output=True, text=True)
check(pstale.returncode != 0 and "stale/incompatible scan" in (pstale.stderr + pstale.stdout),
      "run_bm.py rejects stale scan provenance")

# 4) scan reuse guard; full FEM end-to-end if the optional mesher is installed
from run_bm import load_scan_row  # noqa: E402
rec = load_scan_row(os.path.join(tmp, "scan.jsonl"), 8, 2, 10)
check(rec.get("graph_feature_version") == GRAPH_FEATURE_VERSION and rec.get("scan_schema_version") == SCAN_SCHEMA_VERSION,
      "run_bm.py accepts a current scan record with matching W")

out = os.path.join(tmp, "r.json")
try:
    import triangle  # noqa: F401
    have_triangle = True
except ImportError:
    have_triangle = False

if have_triangle:
    subprocess.run([sys.executable, "run_bm.py", "--n", "8", "--seed", "2", "--h", "0.2", "--neig", "6", "--quiet",
                    "--graph-from", os.path.join(tmp, "scan.jsonl"), "--out", out], check=True, capture_output=True)
    r = json.load(open(out))
    check(r.get("schema_version") == SCHEMA_VERSION, "run_bm.py output has schema_version %s" % r.get("schema_version"))
    check("u_per_cusp" in r["liouville"] and "eps_h_central" in r["liouville"], "liouville block uses u_per_cusp / eps_h_*")
    check(r["graph"]["graph_source"].startswith("scan:"), "graph features reused from the scan JSONL")
    check(r["dtn"].get("certified") is False, "cusp DtN result carries certified=false")
    p = subprocess.run([sys.executable, "analyze.py", "h4", "--glob", out, "--boot", "5"], capture_output=True, text=True)
    check(p.returncode == 0 and "conformal density" in p.stdout and "H4 diagnostic: c0 = F(k-bin)" in p.stdout and "partial R^2" in p.stdout,
          "analyze.py h4 reads the result and reports the c0 regression")
else:
    print("  SKIP full FEM end-to-end (optional package 'triangle' not installed)")
    r = None

# 5) disc formula on this small surface (when FEM dependency is available)
if r is not None:
    errs = []
    for c in r["liouville"]["u_per_cusp"]:
        ual = c["u_at_length"]
        if "1" in ual and "2" in ual:
            u1, _, l1 = ual["1"]; u2, _, l2 = ual["2"]
            rho = 2 / math.sqrt(math.exp(2 * (u1 - math.log(2 * math.pi / l1) + 2 * math.pi / l1)))
            r0 = math.exp(-2 * math.pi / l2)
            errs.append(abs(u2 - math.log((4 * math.pi / l2) * r0 * rho / (rho ** 2 - r0 ** 2))))
    check(bool(errs) and max(errs) < 1e-2, "disc formula predicts u(2) from u(1) within 1e-2 (max err %.1e)" % (max(errs) if errs else float("nan")))


# 6) statistical surface identity: two resolutions of one (n,seed) are one surface
fake = [dict(n=32, seed=7, h=0.10), dict(n=32, seed=7, h=0.07), dict(n=32, seed=8, h=0.10)]
sel = select_unique_surfaces(fake)
check(len(sel) == 2 and min(x["h"] for x in sel if x["seed"] == 7) == 0.07,
      "resolution selection counts (n,seed) once and keeps the finest h")

# 7) H4 regression is genuinely on c0: recover known synthetic c0 coefficients
rng = np.random.default_rng(123)
truth = np.array([0.7, -1.1, 0.35, 0.2])
Csynt = []
sid = 0
for k in (2, 4):
    for j in range(20):
        x = rng.normal(size=4)
        c0 = (4.0 + 0.1 * k) + float(x @ truth)
        Csynt.append(dict(n=64, seed=sid, sid=sid, k=k, c0=c0,
                          feats={f: float(v) for f, v in zip(FEATS, x)}))
        sid += 1
bsynt, _, _ = _regress_c0(Csynt)
check(np.max(np.abs(bsynt - truth)) < 1e-10,
      "H4 residual regression uses c0 and recovers synthetic coefficients")


# 8) aggregate.py must count independent surfaces, not mesh files
agg_tmp = tempfile.mkdtemp()
example_glob = os.path.join(HERE, "results_example", "*.json")
pagg = subprocess.run([sys.executable, os.path.join(HERE, "aggregate.py"), example_glob],
                      cwd=agg_tmp, capture_output=True, text=True)
summary_path = os.path.join(agg_tmp, "summary.csv")
all_path = os.path.join(agg_tmp, "summary_all_resolutions.csv")
summary_rows = open(summary_path).read().strip().splitlines() if os.path.exists(summary_path) else []
all_rows = open(all_path).read().strip().splitlines() if os.path.exists(all_path) else []
check(pagg.returncode == 0 and len(summary_rows) == 15 and len(all_rows) == 16 and "n=32 seed=1" in pagg.stdout,
      "aggregate.py reduces 15 files to 14 independent surfaces while retaining all resolutions separately")

# 9) Archived examples are smoke/regression data, never reusable production results.
valid_example = os.path.join(HERE, "results_example", "bm_n16_s1_h0.12.json")
parch = subprocess.run([sys.executable, os.path.join(HERE, "validate_result.py"), valid_example,
                         "--n", "16", "--seed", "1", "--h", "0.12", "--L0", "1.0", "--T-ext", "6.0", "--neig", "20", "--W", "10"], capture_output=True, text=True)
check(parch.returncode != 0 and "not a native numerical run" in (parch.stdout + parch.stderr),
      "resume validator rejects metadata-migrated archived examples")

# Build a current-schema native fixture for adversarial validator tests.  Its numerical
# values originate from an archived example, but this temporary object is used only to
# exercise validation logic, never as scientific data.
fixture = json.load(open(valid_example))
fixture["code_version"] = __version__
fixture["schema_version"] = SCHEMA_VERSION
fixture["provenance"] = dict(kind="native_run", numerical_payload_recomputed=True, generated_by=__version__)
fixture["dtn"].setdefault("tol", 1e-10)
fixture["dtn"].setdefault("residual", 1e-12)
fixture["quality"] = dict(analysis_eligible=True, compact_analysis_eligible=True,
                          cusped_analysis_eligible=True, h4_analysis_eligible=True,
                          full_analysis_eligible=True)
native_fixture = os.path.join(tmp, "native_fixture.json")
json.dump(fixture, open(native_fixture, "w"), indent=1)

base_cmd = [sys.executable, os.path.join(HERE, "validate_result.py"), native_fixture,
            "--n", "16", "--seed", "1", "--h", "0.12", "--L0", "1.0", "--T-ext", "6.0", "--neig", "20", "--W", "10"]
pvalid = subprocess.run(base_cmd + ["--require", "full"], capture_output=True, text=True)
pbad = subprocess.run(base_cmd[:-8] + ["0.11"] if False else
                      [sys.executable, os.path.join(HERE, "validate_result.py"), native_fixture,
                       "--n", "16", "--seed", "1", "--h", "0.11", "--L0", "1.0", "--T-ext", "6.0", "--neig", "20", "--W", "10"],
                      capture_output=True, text=True)
check(pvalid.returncode == 0 and pbad.returncode != 0,
      "validate_result.py accepts a matching native fixture and rejects parameter mismatches")

# 10) n fixed effects block a spurious size-confounded H4 signal
rng2 = np.random.default_rng(456)
Cconf = []
sid = 0
for nn in (32, 128):
    for j in range(60):
        k = 2 if j % 2 == 0 else 4
        x = rng2.normal(size=4) + (3.0 if nn == 128 else -3.0)
        c0 = 4.0 + 0.08 * k + (0.9 if nn == 128 else -0.9)
        Cconf.append(dict(n=nn, seed=sid, sid=sid, k=k, c0=c0,
                          feats={f: float(v) for f, v in zip(FEATS, x)}))
        sid += 1
bconf, pr2conf, _ = _regress_c0(Cconf)
check(np.max(np.abs(bconf)) < 1e-10 and abs(pr2conf) < 1e-10,
      "H4 n fixed effects remove a purely size-confounded graph-feature signal")

# 11) quarantine files and recomputed quality must protect downstream statistics.
qdir = tempfile.mkdtemp()
good_dst = os.path.join(qdir, "good.json")
json.dump(fixture, open(good_dst, "w"), indent=1)
q_dst = os.path.join(qdir, "bad.invalid.20260917.json")
json.dump(fixture, open(q_dst, "w"), indent=1)
badq = json.loads(json.dumps(fixture)); badq["liouville"]["converged"] = False
badq["quality"]["analysis_eligible"] = True  # malicious/stale stored flag must not matter
nonconv = os.path.join(qdir, "nonconv.json")
json.dump(badq, open(nonconv, "w"), indent=1)
pq = subprocess.run([sys.executable, "analyze.py", "h4", "--glob", os.path.join(qdir, "*.json"), "--boot", "2"], capture_output=True, text=True)
check(pq.returncode == 0 and "excluding 1 invalid" in pq.stdout,
      "analysis ignores quarantined filenames and recomputes quality instead of trusting flags")

# 12) validator rejects NaN compact eigenvalues even with optimistic stored flags.
nanr = json.loads(json.dumps(fixture)); nanr["lambda1_compact"] = float("nan"); nanr["quality"]["analysis_eligible"] = True
nanfile = os.path.join(tmp, "nan.json"); json.dump(nanr, open(nanfile, "w"), allow_nan=True)
pnan = subprocess.run(base_cmd[:2] + [nanfile] + base_cmd[3:] + ["--require", "compact"], capture_output=True, text=True)
check(pnan.returncode != 0, "validator recomputes finiteness and rejects NaN despite optimistic stored quality flags")

# 13) H4 validates the exact profile fields it consumes, including positive horocycle length.
badprof = json.loads(json.dumps(fixture))
first = badprof["liouville"]["u_per_cusp"][0]
key = next(iter(first["u_at_length"]))
first["u_at_length"][key][2] = 0.0
badprof_file = os.path.join(tmp, "badprof.json"); json.dump(badprof, open(badprof_file, "w"))
pbadprof = subprocess.run([sys.executable, "analyze.py", "h4", "--glob", badprof_file, "--boot", "2"], capture_output=True, text=True)
check(pbadprof.returncode != 0 and "invalid for h4" in (pbadprof.stdout + pbadprof.stderr),
      "H4 rejects zero/non-finite horocycle profile inputs before c0 division")

# 14) converged=True is insufficient when residual exceeds the recorded tolerance.
badres = json.loads(json.dumps(fixture)); badres["liouville"]["converged"] = True; badres["liouville"]["residual"] = 1.0; badres["liouville"]["tol"] = 1e-11
badres_file = os.path.join(tmp, "badres.json"); json.dump(badres, open(badres_file, "w"))
pbadres = subprocess.run(base_cmd[:2] + [badres_file] + base_cmd[3:] + ["--require", "compact"], capture_output=True, text=True)
check(pbadres.returncode != 0, "validator cross-checks converged flag against residual <= tolerance")

# 15) duplicate top-level/run_parameters identity fields must agree.
badseed = json.loads(json.dumps(fixture)); badseed["seed"] = 999
badseed_file = os.path.join(tmp, "badseed.json"); json.dump(badseed, open(badseed_file, "w"))
pbadseed = subprocess.run(base_cmd[:2] + [badseed_file] + base_cmd[3:] + ["--require", "compact"], capture_output=True, text=True)
check(pbadseed.returncode != 0 and "disagree" in (pbadseed.stdout + pbadseed.stderr),
      "validator rejects conflicting top-level and run_parameters seed")

# 16) failed DtN invalidates only cusp/full branches; compact remains reusable.
baddtn = json.loads(json.dumps(fixture)); baddtn["dtn"]["converged"] = False; baddtn["dtn"]["status"] = "root_not_converged"; baddtn["dtn"]["residual"] = 1.0
baddtn["lambda1_cusped"] = None; baddtn["lambda1_cusped_raw"] = 0.2
baddtn_file = os.path.join(tmp, "baddtn.json"); json.dump(baddtn, open(baddtn_file, "w"))
pc = subprocess.run(base_cmd[:2] + [baddtn_file] + base_cmd[3:] + ["--require", "compact"], capture_output=True, text=True)
pf = subprocess.run(base_cmd[:2] + [baddtn_file] + base_cmd[3:] + ["--require", "full"], capture_output=True, text=True)
check(pc.returncode == 0 and pf.returncode != 0,
      "failed DtN solve rejects cusp/full branches, not the independently valid compact branch")

# 17) CSV export blanks invalid cusp values and exports recomputed branch status.
csvdir = tempfile.mkdtemp()
subprocess.run([sys.executable, os.path.join(HERE, "aggregate.py"), baddtn_file], cwd=csvdir, capture_output=True, text=True, check=True)
import csv
with open(os.path.join(csvdir, "summary.csv"), newline="") as fh:
    rr = next(csv.DictReader(fh))
check(rr["lambda1_cusped"] == "" and rr["delta_compact_minus_cusped"] == "" and
      rr["cusped_analysis_eligible"] == "False" and rr["compact_analysis_eligible"] == "True",
      "CSV blanks invalid cusp values and exports branch-specific quality status")

# 18) W remains part of resume provenance.
pbadW = subprocess.run([sys.executable, os.path.join(HERE, "validate_result.py"), native_fixture,
                       "--n", "16", "--seed", "1", "--h", "0.12", "--L0", "1.0", "--T-ext", "6.0", "--neig", "20", "--W", "8"], capture_output=True, text=True)
check(pbadW.returncode != 0, "validate_result.py rejects changed W / computation provenance")


# 19) compact lambda1 must be exactly the first positive entry of eigs_compact.
badlam = json.loads(json.dumps(fixture))
badlam["lambda1_compact"] = float(badlam["eigs_compact"][1]) + 0.05
badlam_file = os.path.join(tmp, "bad_lambda1_compact.json"); json.dump(badlam, open(badlam_file, "w"))
pbadlam = subprocess.run(base_cmd[:2] + [badlam_file] + base_cmd[3:] + ["--require", "compact"], capture_output=True, text=True)
check(pbadlam.returncode != 0,
      "validator rejects lambda1_compact inconsistent with eigs_compact[1]")

# 20) compact spectrum must contain the requested neig+1 values (including zero mode).
short = json.loads(json.dumps(fixture))
short["eigs_compact"] = short["eigs_compact"][:2]
short_file = os.path.join(tmp, "short_eigs_compact.json"); json.dump(short, open(short_file, "w"))
pshort = subprocess.run(base_cmd[:2] + [short_file] + base_cmd[3:] + ["--require", "compact"], capture_output=True, text=True)
check(pshort.returncode != 0,
      "validator rejects a compact eigenvalue list shorter than run_parameters.neig + 1")

# 21) Liouville convergence metadata must be persisted as the solver reports it.
# This catches the 0.4.4 regression where run_bm dropped converged/status while
# result_quality correctly required them.
run_src = open(os.path.join(HERE, "run_bm.py")).read()
check('converged=bool(liou.get("converged", False))' in run_src and
      'status=str(liou.get("status", "unknown"))' in run_src,
      "run_bm.py persists actual Liouville converged/status metadata")

print("\n%s" % ("ALL CHECKS PASSED -- this directory holds bm_cusps %s" % __version__ if fails == 0
                else "%d CHECK(S) FAILED -- do not launch the campaign from this directory" % fails))
sys.exit(1 if fails else 0)
