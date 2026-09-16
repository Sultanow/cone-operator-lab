#!/usr/bin/env python3
"""Validate an existing run_bm.py result before a SLURM task is skipped.

Exit 0 only when the file is current and matches the requested (n, seed, h).
Anything stale, truncated, mismatched, or structurally incomplete exits non-zero.
"""
import argparse
import json
import math
import os
import sys

from version import GRAPH_FEATURE_VERSION, SCHEMA_VERSION


def fail(msg):
    print("INVALID: " + msg, file=sys.stderr)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--h", type=float, required=True)
    a = ap.parse_args()
    if not os.path.isfile(a.file) or os.path.getsize(a.file) == 0:
        return fail("missing or empty file")
    try:
        with open(a.file) as f:
            r = json.load(f)
    except Exception as e:
        return fail("not valid JSON: %s" % e)
    if r.get("schema_version") != SCHEMA_VERSION:
        return fail("schema_version=%r, expected %d" % (r.get("schema_version"), SCHEMA_VERSION))
    if int(r.get("n", -1)) != a.n or int(r.get("seed", -1)) != a.seed:
        return fail("parameter mismatch: result has n=%r seed=%r" % (r.get("n"), r.get("seed")))
    try:
        if not math.isclose(float(r.get("h")), a.h, rel_tol=0.0, abs_tol=1e-12):
            return fail("h mismatch: result has %r, requested %.17g" % (r.get("h"), a.h))
    except Exception:
        return fail("missing/invalid h")
    required = ("V", "genus", "lambda1_compact", "liouville", "weyl", "graph")
    miss = [k for k in required if k not in r]
    if miss:
        return fail("missing required keys: %s" % ", ".join(miss))
    li = r.get("liouville", {})
    if not isinstance(li.get("u_per_cusp"), list) or "eps_h_central" not in li:
        return fail("incomplete liouville block")
    g = r.get("graph", {})
    if g.get("graph_feature_version") != GRAPH_FEATURE_VERSION:
        return fail("graph_feature_version=%r, expected %d" % (g.get("graph_feature_version"), GRAPH_FEATURE_VERSION))
    if not g.get("graph_source"):
        return fail("missing graph provenance")
    print("VALID: %s (schema=%d graph_feature=%d n=%d seed=%d h=%g)" %
          (a.file, SCHEMA_VERSION, GRAPH_FEATURE_VERSION, a.n, a.seed, a.h))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
