"""Central, re-computed quality checks for Brooks--Makover result JSON.

Stored quality flags are descriptive only.  Every consumer must call
``recompute_quality`` and base acceptance on the returned values.
"""
from __future__ import annotations
import math


def _finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _finite_seq(xs, min_len=0):
    return isinstance(xs, list) and len(xs) >= min_len and all(_finite(x) for x in xs)


def _seed_consistent(r):
    rp = r.get("run_parameters")
    if not isinstance(rp, dict):
        return False
    return r.get("seed") == rp.get("seed") and r.get("n") == rp.get("n")


def _liouville_ok(r):
    li = r.get("liouville")
    if not isinstance(li, dict) or li.get("converged") is not True:
        return False
    residual, tol = li.get("residual"), li.get("tol")
    # Current native runs must carry both quantities.  Historical migrated examples
    # may not; consumers can inspect provenance and decide whether to use them.
    if not (_finite(residual) and _finite(tol) and float(tol) > 0):
        return False
    return float(residual) <= float(tol)


def _compact_eigs_ok(r):
    """Validate the compact eigenvalue payload, not merely its finiteness.

    Native run_bm writes neig+1 eigenvalues because index 0 is the constant
    zero mode and lambda1_compact is eigs_compact[1].  Both invariants are
    checked here so truncated/corrupted cached results cannot pass quality.
    """
    eigs = r.get("eigs_compact")
    lam1 = r.get("lambda1_compact")
    rp = r.get("run_parameters") if isinstance(r.get("run_parameters"), dict) else {}
    try:
        neig = int(rp.get("neig"))
    except Exception:
        return False
    if neig < 1 or not _finite_seq(eigs, neig + 1):
        return False
    if not (_finite(lam1) and float(lam1) > 0):
        return False
    if not math.isclose(float(lam1), float(eigs[1]), rel_tol=1e-10, abs_tol=1e-12):
        return False
    return True


def _profile_ok(r):
    """Validate exactly the fields used by H4/c0_table."""
    li = r.get("liouville")
    if not isinstance(li, dict) or not isinstance(li.get("u_per_cusp"), list):
        return False
    if not li["u_per_cusp"]:
        return False
    for p in li["u_per_cusp"]:
        if not isinstance(p, dict):
            return False
        try:
            if int(p.get("k", 0)) <= 0:
                return False
        except Exception:
            return False
        ual = p.get("u_at_length")
        if not isinstance(ual, dict) or not ual:
            return False
        usable = False
        for _, trip in ual.items():
            if not (isinstance(trip, (list, tuple)) and len(trip) == 3):
                return False
            mean_u, sup_u, actual_length = trip
            if not (_finite(mean_u) and _finite(sup_u) and _finite(actual_length)):
                return False
            if float(actual_length) <= 0:
                return False
            usable = True
        if not usable:
            return False
    return True


def _dtn_ok(r):
    d = r.get("dtn")
    if not isinstance(d, dict) or d.get("converged") is not True:
        return False
    status = d.get("status")
    if status == "no_l2_detected":
        return r.get("lambda1_cusped") is None
    if status != "root_converged":
        return False
    lam = r.get("lambda1_cusped")
    if not (_finite(lam) and 0 < float(lam) < 0.25):
        return False
    residual, tol = d.get("residual"), d.get("tol")
    if not (_finite(residual) and _finite(tol) and float(tol) > 0):
        return False
    return float(residual) <= float(tol)


def recompute_quality(r):
    seed_ok = _seed_consistent(r)
    li_ok = _liouville_ok(r)
    ce_ok = _compact_eigs_ok(r)
    prof_ok = _profile_ok(r)
    compact_ok = seed_ok and li_ok and ce_ok
    cusp_ok = seed_ok and _dtn_ok(r)
    return {
        "identity_consistent": bool(seed_ok),
        "liouville_converged": bool(li_ok),
        "compact_eigs_finite": bool(ce_ok),
        "h4_profile_valid": bool(prof_ok),
        "compact_analysis_eligible": bool(compact_ok),
        "h4_analysis_eligible": bool(compact_ok and prof_ok),
        "cusped_analysis_eligible": bool(cusp_ok),
        "full_analysis_eligible": bool(compact_ok and cusp_ok),
        # Backward-compatible alias used by H1 and compact-surface ensemble code.
        "analysis_eligible": bool(compact_ok),
    }


def provenance_is_native(r):
    p = r.get("provenance", {})
    return isinstance(p, dict) and p.get("kind") == "native_run" and p.get("numerical_payload_recomputed") is True
