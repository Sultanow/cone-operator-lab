#!/usr/bin/env python3
"""Validate an existing run_bm.py result before a SLURM task is skipped.

Exit 0 only when the file uses the current schema, matches every numerical
parameter that changes the computation, and is marked analysis-eligible by the
solver convergence policy.  Stale, truncated, mismatched, or non-converged
results exit non-zero and must not be silently reused.
"""
import argparse, json, math, os, sys
from version import GRAPH_FEATURE_VERSION, SCHEMA_VERSION, __version__

def fail(msg):
    print("INVALID: " + msg, file=sys.stderr); return 1

def same_float(x, y):
    try: return math.isclose(float(x), float(y), rel_tol=0.0, abs_tol=1e-12)
    except Exception: return False

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("file")
    ap.add_argument("--n",type=int,required=True); ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--h",type=float,required=True); ap.add_argument("--L0",type=float,required=True)
    ap.add_argument("--T-ext",dest="T_ext",type=float,required=True)
    ap.add_argument("--neig",type=int,required=True); ap.add_argument("--W",type=int,required=True)
    a=ap.parse_args()
    if not os.path.isfile(a.file) or os.path.getsize(a.file)==0: return fail("missing or empty file")
    try: r=json.load(open(a.file))
    except Exception as e: return fail("not valid JSON: %s"%e)
    if r.get("schema_version") != SCHEMA_VERSION: return fail("schema_version=%r, expected %d"%(r.get("schema_version"),SCHEMA_VERSION))
    if r.get("code_version") != __version__: return fail("code_version=%r, expected %s"%(r.get("code_version"),__version__))
    rp=r.get("run_parameters")
    if not isinstance(rp,dict): return fail("missing run_parameters provenance")
    expected={"n":a.n,"seed":a.seed,"h":a.h,"L0":a.L0,"T_ext":a.T_ext,"neig":a.neig,"W":a.W}
    for k,v in expected.items():
        got=rp.get(k)
        if isinstance(v,float):
            if not same_float(got,v): return fail("%s mismatch: result has %r, requested %.17g"%(k,got,v))
        elif got != v: return fail("%s mismatch: result has %r, requested %r"%(k,got,v))
    required=("V","genus","lambda1_compact","liouville","weyl","graph","quality")
    miss=[k for k in required if k not in r]
    if miss: return fail("missing required keys: %s"%", ".join(miss))
    li=r.get("liouville",{})
    if not isinstance(li.get("u_per_cusp"),list) or "eps_h_central" not in li: return fail("incomplete liouville block")
    if not li.get("converged",False): return fail("Liouville solve is not converged")
    q=r.get("quality",{})
    if not q.get("analysis_eligible",False) or not q.get("compact_solver_converged",False): return fail("result is not analysis-eligible / compact solve did not converge")
    g=r.get("graph",{})
    if g.get("graph_feature_version") != GRAPH_FEATURE_VERSION: return fail("graph_feature_version=%r, expected %d"%(g.get("graph_feature_version"),GRAPH_FEATURE_VERSION))
    if int(g.get("W",-1)) != a.W: return fail("graph W=%r, expected %d"%(g.get("W"),a.W))
    if not g.get("graph_source"): return fail("missing graph provenance")
    print("VALID: %s (schema=%d code=%s graph_feature=%d n=%d seed=%d h=%g L0=%g T_ext=%g neig=%d W=%d)" %
          (a.file,SCHEMA_VERSION,__version__,GRAPH_FEATURE_VERSION,a.n,a.seed,a.h,a.L0,a.T_ext,a.neig,a.W))
    return 0
if __name__=="__main__": raise SystemExit(main())
