"""Recompute result eligibility from numerical payloads; never trust stored flags alone."""
import math


def _finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _finite_seq(xs, min_len=0):
    return isinstance(xs, list) and len(xs) >= min_len and all(_finite(x) for x in xs)


def assess_result(r):
    """Return independently derived compact/cusped/full eligibility and reasons.

    This deliberately ignores r['quality'] while deriving the result.  Callers may
    separately compare the stored flags with this assessment to detect corruption.
    """
    reasons = {"compact": [], "cusped": []}

    # Compact branch ---------------------------------------------------------
    li = r.get("liouville", {}) if isinstance(r.get("liouville"), dict) else {}
    eigc = r.get("eigs_compact")
    lamc = r.get("lambda1_compact")
    if li.get("converged") is not True:
        reasons["compact"].append("Liouville solve not converged")
    for key in ("residual", "area_compact", "area_compact_exact", "chi_discrete"):
        if not _finite(li.get(key)):
            reasons["compact"].append("non-finite liouville.%s" % key)
    if not _finite(lamc):
        reasons["compact"].append("lambda1_compact is non-finite")
    if not _finite_seq(eigc, 2):
        reasons["compact"].append("eigs_compact missing/contains non-finite values")
    elif _finite(lamc) and not math.isclose(float(lamc), float(eigc[1]), rel_tol=1e-10, abs_tol=1e-12):
        reasons["compact"].append("lambda1_compact disagrees with eigs_compact[1]")
    weyl = r.get("weyl", {}) if isinstance(r.get("weyl"), dict) else {}
    if not _finite(weyl.get("slope")):
        reasons["compact"].append("non-finite Weyl slope")
    # H4 consumes per-cusp profiles, so they are part of compact eligibility.
    upc = li.get("u_per_cusp")
    if not isinstance(upc, list) or not upc:
        reasons["compact"].append("missing liouville.u_per_cusp")
    else:
        for i, p in enumerate(upc):
            if not isinstance(p, dict) or not _finite(p.get("u_height1_mean")) or not _finite(p.get("u_height1_sup")):
                reasons["compact"].append("non-finite cusp profile at index %d" % i)
                break

    compact_ok = not reasons["compact"]

    # Cusped/DtN branch ------------------------------------------------------
    dtn = r.get("dtn", {}) if isinstance(r.get("dtn"), dict) else {}
    status = dtn.get("status")
    dtn_conv = dtn.get("converged") is True
    lam = r.get("lambda1_cusped")
    raw = r.get("lambda1_cusped_raw", lam)
    if not dtn_conv:
        reasons["cusped"].append("DtN solve not converged")
    elif status == "root_converged":
        if not _finite(lam) or not (0.0 < float(lam) < 0.25):
            reasons["cusped"].append("root_converged but accepted cusp eigenvalue is not finite in (0,1/4)")
        if not _finite(raw):
            reasons["cusped"].append("root_converged but raw cusp eigenvalue is non-finite")
        elif _finite(lam) and not math.isclose(float(raw), float(lam), rel_tol=1e-10, abs_tol=1e-12):
            reasons["cusped"].append("accepted and raw cusp eigenvalues disagree")
    elif status == "no_l2_detected":
        if lam is not None:
            reasons["cusped"].append("no_l2_detected must have lambda1_cusped=null")
        if raw is not None:
            reasons["cusped"].append("no_l2_detected must have lambda1_cusped_raw=null")
    else:
        reasons["cusped"].append("unrecognized/failed DtN status %r" % status)

    cusped_ok = not reasons["cusped"]
    return dict(
        compact_analysis_eligible=compact_ok,
        cusped_analysis_eligible=cusped_ok,
        full_analysis_eligible=bool(compact_ok and cusped_ok),
        reasons=reasons,
    )


def stored_quality_consistent(r, derived):
    q = r.get("quality", {}) if isinstance(r.get("quality"), dict) else {}
    expected = {
        "analysis_eligible": derived["compact_analysis_eligible"],
        "compact_analysis_eligible": derived["compact_analysis_eligible"],
        "cusped_analysis_eligible": derived["cusped_analysis_eligible"],
        "full_analysis_eligible": derived["full_analysis_eligible"],
    }
    bad = []
    for k, v in expected.items():
        if q.get(k) is not v:
            bad.append("quality.%s=%r, derived=%r" % (k, q.get(k), v))
    return bad
