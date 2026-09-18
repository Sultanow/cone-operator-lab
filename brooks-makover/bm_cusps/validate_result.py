#!/usr/bin/env python3
"""Validate a run_bm.py result before reuse.

Stored quality flags are not trusted. Eligibility is reconstructed from the numerical
payload, and native numerical provenance is required for production resume.
"""
import argparse, json, math, os, sys
from version import GRAPH_FEATURE_VERSION, SCHEMA_VERSION, __version__
from result_quality import assess_result, stored_quality_consistent


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
    ap.add_argument("--require", choices=("compact","cusped","full"), default="full",
                    help="which independently reconstructed branch(es) must be eligible")
    a=ap.parse_args()
    if not os.path.isfile(a.file) or os.path.getsize(a.file)==0: return fail("missing or empty file")
    try: r=json.load(open(a.file))
    except Exception as e: return fail("not valid JSON: %s"%e)
    if r.get("schema_version") != SCHEMA_VERSION: return fail("schema_version=%r, expected %d"%(r.get("schema_version"),SCHEMA_VERSION))

    prov=r.get("provenance",{})
    if not isinstance(prov,dict) or prov.get("kind") != "native_run" or prov.get("numerical_payload_recomputed") is not True or prov.get("numerical_payload_code_version") != __version__ or prov.get("metadata_migrated") is not False:
        return fail("numerical payload is not a native current-version run; archived/migrated examples are never reusable production results")
    if r.get("code_version") != __version__: return fail("code_version=%r, expected %s"%(r.get("code_version"),__version__))

    rp=r.get("run_parameters")
    if not isinstance(rp,dict): return fail("missing run_parameters provenance")
    expected={"n":a.n,"seed":a.seed,"h":a.h,"L0":a.L0,"T_ext":a.T_ext,"neig":a.neig,"W":a.W}
    for k,v in expected.items():
        got=rp.get(k)
        if isinstance(v,float):
            if not same_float(got,v): return fail("%s mismatch: result has %r, requested %.17g"%(k,got,v))
        elif got != v: return fail("%s mismatch: result has %r, requested %r"%(k,got,v))

    required=("V","genus","lambda1_compact","eigs_compact","liouville","weyl","graph","quality","provenance","dtn")
    miss=[k for k in required if k not in r]
    if miss: return fail("missing required keys: %s"%", ".join(miss))
    def finite(x):
        try: return math.isfinite(float(x))
        except Exception: return False
    for key in ("lambda1_thick", "lambda1_neumann", "lambda1_compact"):
        if not finite(r.get(key)): return fail("%s is missing or non-finite" % key)
    for key, minlen in (("eigs_thick",2),("eigs_neumann",2),("eigs_compact",a.neig+1)):
        xs=r.get(key)
        if not isinstance(xs,list) or len(xs)<minlen or not all(finite(x) for x in xs):
            return fail("%s is too short or contains non-finite values" % key)

    g=r.get("graph",{})
    if g.get("graph_feature_version") != GRAPH_FEATURE_VERSION: return fail("graph_feature_version=%r, expected %d"%(g.get("graph_feature_version"),GRAPH_FEATURE_VERSION))
    if int(g.get("W",-1)) != a.W: return fail("graph W=%r, expected %d"%(g.get("W"),a.W))
    if not g.get("graph_source"): return fail("missing graph provenance")

    derived=assess_result(r)
    inconsistent=stored_quality_consistent(r,derived)
    if inconsistent: return fail("stored quality flags contradict numerical payload: " + "; ".join(inconsistent))
    if a.require == "compact" and not derived["compact_analysis_eligible"]:
        return fail("compact branch rejected: " + "; ".join(derived["reasons"]["compact"]))
    if a.require == "cusped" and not derived["cusped_analysis_eligible"]:
        return fail("cusped/DtN branch rejected: " + "; ".join(derived["reasons"]["cusped"]))
    if a.require == "full" and not derived["full_analysis_eligible"]:
        rs=derived["reasons"]["compact"]+derived["reasons"]["cusped"]
        return fail("full result rejected: " + "; ".join(rs))

    print("VALID: %s (require=%s schema=%d code=%s graph_feature=%d n=%d seed=%d h=%g L0=%g T_ext=%g neig=%d W=%d)" %
          (a.file,a.require,SCHEMA_VERSION,__version__,GRAPH_FEATURE_VERSION,a.n,a.seed,a.h,a.L0,a.T_ext,a.neig,a.W))
    return 0
if __name__=="__main__": raise SystemExit(main())
