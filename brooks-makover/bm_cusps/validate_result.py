#!/usr/bin/env python3
"""Validate a cached run_bm.py result before reuse.

Quality is recomputed from the numerical payload; stored quality flags are never trusted.
Use --require compact|cusped|h4|full to state which scientific branch must be valid.
"""
import argparse, json, math, os, sys
from version import GRAPH_FEATURE_VERSION, SCHEMA_VERSION, __version__
from result_quality import recompute_quality, provenance_is_native


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
    ap.add_argument("--require", choices=("compact","cusped","h4","full"), default="compact")
    a=ap.parse_args()
    if not os.path.isfile(a.file) or os.path.getsize(a.file)==0: return fail("missing or empty file")
    try: r=json.load(open(a.file))
    except Exception as e: return fail("not valid JSON: %s"%e)
    if r.get("schema_version") != SCHEMA_VERSION: return fail("schema_version=%r, expected %d"%(r.get("schema_version"),SCHEMA_VERSION))
    if r.get("code_version") != __version__: return fail("code_version=%r, expected %s"%(r.get("code_version"),__version__))
    if not provenance_is_native(r): return fail("result is not a native numerical run of the current pipeline")
    rp=r.get("run_parameters")
    if not isinstance(rp,dict): return fail("missing run_parameters provenance")
    expected={"n":a.n,"seed":a.seed,"h":a.h,"L0":a.L0,"T_ext":a.T_ext,"neig":a.neig,"W":a.W}
    for k,v in expected.items():
        got=rp.get(k)
        if isinstance(v,float):
            if not same_float(got,v): return fail("%s mismatch: result has %r, requested %.17g"%(k,got,v))
        elif got != v: return fail("%s mismatch: result has %r, requested %r"%(k,got,v))
    if r.get("seed") != rp.get("seed") or r.get("n") != rp.get("n"):
        return fail("top-level n/seed disagree with run_parameters")
    required=("V","genus","lambda1_compact","liouville","weyl","graph","quality","dtn")
    miss=[k for k in required if k not in r]
    if miss: return fail("missing required keys: %s"%", ".join(miss))
    g=r.get("graph",{})
    if g.get("graph_feature_version") != GRAPH_FEATURE_VERSION: return fail("graph_feature_version=%r, expected %d"%(g.get("graph_feature_version"),GRAPH_FEATURE_VERSION))
    if int(g.get("W",-1)) != a.W: return fail("graph W=%r, expected %d"%(g.get("W"),a.W))
    if not g.get("graph_source"): return fail("missing graph provenance")

    q=recompute_quality(r)
    key={"compact":"compact_analysis_eligible","cusped":"cusped_analysis_eligible",
         "h4":"h4_analysis_eligible","full":"full_analysis_eligible"}[a.require]
    if not q[key]:
        return fail("recomputed quality rejects %s branch: %s" % (a.require, q))
    print("VALID: %s (require=%s schema=%d code=%s n=%d seed=%d h=%g)" %
          (a.file,a.require,SCHEMA_VERSION,__version__,a.n,a.seed,a.h))
    return 0
if __name__=="__main__": raise SystemExit(main())
