#!/usr/bin/env python3
"""
run_bm.py -- one Brooks-Makover sample: cusped surface S vs. its compactification S^bar.

Per sample (JSON):
  combinatorics    n, V (cusps), cusp lengths k_c, genus g, chi
  lambda1_thick    first eigenvalues of S_Y (cusps cut at horocycles of length L0, Neumann)
  lambda1_neumann  same with cusps extended by T_ext (Neumann far out in the cusps)
  lambda1_cusped   smallest L^2 eigenvalue of S below 1/4 (exact cusp DtN), or null
  lambda1_compact  first eigenvalues of the uniformized compactification S^bar
  liouville        Newton diagnostics, discrete Gauss-Bonnet, conformal factor u per cusp
                   (mean and sup on the height-1 horocycle) and eps_central = sup|u| outside
                   the height-1 horoball neighbourhoods (bi-Lipschitz constant of g vs g_bar there)
  weyl             slope of the counting function of S^bar (Weyl: Area/4pi = g-1)
  graph            combinatorial stage: cycles, tangles, adjacency / non-backtracking spectra,
                   cusp statistics, combinatorial length spectrum (see bmgraph.py)

Usage:  python run_bm.py --n 16 --seed 3 --h 0.1 --L0 1.0 --neig 40 --out results/x.json
"""
import argparse
import json
import math
import sys
import time

import numpy as np

from bmsurf import BMSurface, build_mesh
from bmgraph import graph_features
from bmfem import (CuspDtN, FOUR_PI, curvature_source, cusp_eigs_dtn, lumped, mass,
                   neumann_eigs, lowest, restrict, solve_liouville, stiffness, weyl_fit)

T0 = time.time()
VERBOSE = True


def log(msg):
    if VERBOSE:
        print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


def load_scan_row(path, n, seed):
    """Full graph_features() record for (n, seed) from a graph_scan.py JSONL file
    (a .csv path is mapped to its sibling .jsonl).  None if absent, so the schema of the
    'graph' block is identical whether it is reused or recomputed."""
    import os
    if path.endswith(".csv"):
        path = path[:-4] + ".jsonl"
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("n") == n and rec.get("seed") == seed:
                rec.pop("seed", None)
                rec["simple_cycles"] = {int(k): v for k, v in rec["simple_cycles"].items()}
                rec["tangle_walks"] = {int(k): v for k, v in rec["tangle_walks"].items()}
                return rec
    return None


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="2n ideal triangles")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h", type=float, default=0.1, help="hyperbolic mesh size")
    ap.add_argument("--L0", type=float, default=1.0, help="cusp cut-off horocycle length")
    ap.add_argument("--T-ext", type=float, default=6.0, help="depth of the cusp extension (DtN)")
    ap.add_argument("--neig", type=int, default=40, help="eigenvalues of S^bar for the Weyl fit")
    ap.add_argument("--W", type=int, default=10, help="max word length for cycles / length spectrum")
    ap.add_argument("--graph-from", type=str, default=None,
                    help="scan .jsonl (or .csv -> sibling .jsonl) from graph_scan.py; reuse the full record for (n, seed)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    VERBOSE = not args.quiet

    rng = np.random.default_rng(args.seed)
    surf = BMSurface.random(args.n, rng)
    log("surface: n=%d (%d triangles)  V=%d cusps  g=%d  chi=%d  k=%s  (attempts %d)"
        % (surf.n, 2 * surf.n, surf.V, surf.g, surf.chi, sorted(surf.k.tolist(), reverse=True), surf.attempts))
    if surf.g < 2:
        log("genus < 2: compactification is not hyperbolic, aborting")
        sys.exit(2)
    gf = load_scan_row(args.graph_from, surf.n, args.seed) if args.graph_from else None
    if gf is not None:
        assert gf["V"] == surf.V and gf["genus"] == surf.g, "scan row does not match the generated surface"
        gf["graph_source"] = "scan:%s" % args.graph_from
    else:
        gf = graph_features(surf, W=args.W)
        gf["graph_source"] = "computed"
    log("graph [%s]: girth=%s cycles<=6=%s tangles=%s systole=%s mu2=%.4f nb_rho2=%.4f Ramanujan(adj)=%s"
        % (gf["graph_source"], gf.get("girth"), {k: v for k, v in gf["simple_cycles"].items() if int(k) <= 6},
           gf.get("n_tangle_walks"), gf.get("systole"), gf.get("mu2", float("nan")), gf.get("nb_rho2", float("nan")),
           gf.get("ramanujan_adj")))

    mesh = build_mesh(surf, h=args.h, L0=args.L0, T_ext=args.T_ext)
    nn, tag = mesh.nnodes, mesh.tag
    log("mesh: %d nodes, %d elements (%d S_Y, %d ext, %d cap); cusp cut-offs Y in [%.2f, %.2f]"
        % (nn, len(tag), (tag == 0).sum(), (tag == 1).sum(), (tag == 2).sum(),
           min(mesh.info["Y"]), max(mesh.info["Y"])))

    def build(mask, nodes, phi=None):
        K = stiffness(mesh.tri[mask], mesh.xy[mask], nn)
        M = mass(mesh.tri[mask], mesh.xy[mask], mesh.kind[mask], mesh.rho_const[mask], nn, phi)
        return K, M

    # ---- thick part S_Y (Neumann on the length-L0 horocycles) ------------------
    nY = mesh.nodes_SY()
    K, M = build(tag == 0, nY)
    area_sy = float(lumped(M).sum())
    area_sy_exact = 2 * surf.n * math.pi - sum(mesh.info["L"])
    log("area(S_Y): discrete %.6f  exact %.6f  (rel. err %.2e)"
        % (area_sy, area_sy_exact, abs(area_sy - area_sy_exact) / area_sy_exact))
    vals_T, _ = neumann_eigs(restrict(K, nY), restrict(M, nY), 6)
    log("thick S_Y Neumann: lam_1=%.6f  lam_2=%.6f" % (vals_T[1], vals_T[2]))

    # ---- cusped surface S: extension + Neumann, then exact DtN -------------------
    nS = mesh.nodes_S()
    loc = -np.ones(nn, dtype=int); loc[nS] = np.arange(len(nS))
    K, M = build(tag <= 1, nS)
    K_s, M_s = restrict(K, nS), restrict(M, nS)
    vals_N, _ = neumann_eigs(K_s, M_s, 6)
    log("S (ext, Neumann): lam_1=%.6f  lam_2=%.6f" % (vals_N[1], vals_N[2]))
    dtn = CuspDtN(mesh.dtn_chains, loc, len(nS))
    lam_c, dtn_info = cusp_eigs_dtn(K_s, M_s, dtn, vals_N[1], log=log)
    log("S (exact cusp DtN): lambda_1 = %s   %s" % (lam_c, dtn_info.get("note", "")))

    # ---- compactification S^bar: Liouville, then eigenvalues ---------------------
    nB = mesh.nodes_Sbar()
    locB = -np.ones(nn, dtype=int); locB[nB] = np.arange(len(nB))
    K0, M0 = build(tag != 1, nB)                                 # g_0: hyperbolic on S_Y, flat caps
    _, M_sy = build(tag == 0, nB)
    c = curvature_source(mesh, lumped(M_sy), nn)
    chi_discrete = c[nB].sum() / (2 * math.pi)
    K_b, c_b, ML_b = restrict(K0, nB), c[nB], lumped(M0)[nB]
    phi_b, liou = solve_liouville(K_b, c_b, ML_b, log=log)
    phi = np.zeros(nn); phi[nB] = phi_b
    _, M_g = build(tag != 1, nB, phi=phi)
    M_gb = restrict(M_g, nB)
    area_g = float(lumped(M_gb).sum())
    area_g_exact = FOUR_PI * (surf.g - 1)
    # u = conformal factor, g_bar = e^{2u} g_0 (phi is reserved for eigenfunctions in the papers).
    # For a P1 function the sup over a region is attained at nodes, so max|u| over the central
    # regions is exactly the constant eps in  e^{-2 eps} g <= g_bar <= e^{2 eps} g  on the
    # complement of the height-1 horoball neighbourhoods of all cusps.
    # Per cusp: u on the horocycle of HEIGHT 1 (length k_c, the maximal embedded one) and on the
    # horocycles of fixed LENGTH ell (height k_c/ell), the natural parametrisation for a comparison
    # lemma "outside horoballs of length ell".  eps_outside(ell) = sup|u| off all length-ell
    # horoballs (cusps with k < ell have none: their whole neighbourhood incl. cap counts as
    # outside); eps_long(ell) = the same with those short ("bad") cusps cut out at height 1.
    ELLS = (1, 2, 4, 8, 16, 32, 64)
    u_cusp = []
    outside = {ell: set(mesh.central_nodes.tolist()) for ell in ELLS}
    longonly = {ell: set(mesh.central_nodes.tolist()) for ell in ELLS}
    for ch in mesh.cap_chains:
        c, k, rows = ch.cusp, ch.k, mesh.cusp_rows[ch.cusp]
        h1 = phi[mesh.height1_rows[c]]
        rec = dict(cusp=int(c), k=int(k),
                   u_height1_mean=float(h1.mean()), u_height1_sup=float(np.abs(h1).max()),
                   u_cap_boundary_mean=float(phi[ch.node].mean()), u_at_length={})
        for ell in ELLS:
            if ell <= k:
                j = min(len(rows) - 1, max(0, int(round(math.log(k / ell) / args.h))))
                vals = phi[rows[j]]
                ell_act = k / math.exp(j * args.h)          # the row's actual horocycle length
                rec["u_at_length"][str(ell)] = [float(vals.mean()), float(np.abs(vals).max()), ell_act]
                ids = np.concatenate(rows[:j + 1])
                outside[ell].update(ids.tolist()); longonly[ell].update(ids.tolist())
            else:                                       # no embedded horoball of length ell:
                outside[ell].update(np.concatenate(rows).tolist())   # whole strip counts as outside
                outside[ell].update(ch.node.tolist())                 # (cap interior: see below)
        u_cusp.append(rec)
    for ch in mesh.cap_chains:                          # cap interiors of short cusps are outside too
        for ell in ELLS:
            if ell > ch.k:
                outside[ell].update(mesh.cap_interior_of[ch.cusp].tolist())
    eps_central = float(np.abs(phi[mesh.central_nodes]).max())
    eps_outside = {str(ell): float(np.abs(phi[np.array(sorted(outside[ell]))]).max()) for ell in ELLS}
    eps_long = {str(ell): float(np.abs(phi[np.array(sorted(longonly[ell]))]).max()) for ell in ELLS}
    log("Liouville: %d its, |R|=%.1e ; area(S^bar) %.6f vs 4pi(g-1)=%.6f ; sum c/2pi=%.5f vs chi=%d"
        % (liou["iterations"], liou["residual"], area_g, area_g_exact, chi_discrete, surf.chi))
    log("u on height-1 horocycles (mean): " + "  ".join("k=%d:%.3f" % (p["k"], p["u_height1_mean"]) for p in u_cusp))
    log("eps_central = %.4f ; eps_long(ell) = %s" % (eps_central, {e: round(v, 3) for e, v in eps_long.items()}))
    vals_C, _ = lowest(K_b, M_gb, args.neig + 1, sigma=-0.02)
    log("S^bar: lam_1=%.6f  lam_2=%.6f  lam_3=%.6f" % (vals_C[1], vals_C[2], vals_C[3]))
    weyl = weyl_fit(vals_C, area_expected=area_g_exact)
    log("Weyl slope %.3f (g-1 = %d) -> heard genus %.2f" % (weyl["slope"], surf.g - 1, weyl["heard_genus"]))

    res = dict(
        n=surf.n, seed=args.seed, h=args.h, L0=args.L0, T_ext=args.T_ext,
        V=int(surf.V), genus=int(surf.g), chi=int(surf.chi),
        cusp_lengths=sorted(surf.k.tolist(), reverse=True),
        min_cusp_length=int(surf.k.min()), max_cusp_length=int(surf.k.max()),
        cusp_cutoff_Y=mesh.info["Y"], cusp_horocycle_L=mesh.info["L"],
        mesh=dict(nodes=int(nn), elements=int(len(tag)), ext_elements=int((tag == 1).sum()),
                  cap_elements=int((tag == 2).sum()), area_SY=area_sy, area_SY_exact=area_sy_exact),
        lambda1_thick=float(vals_T[1]), eigs_thick=vals_T.tolist(),
        lambda1_neumann=float(vals_N[1]), eigs_neumann=vals_N.tolist(),
        lambda1_cusped=(None if lam_c is None else float(lam_c)), dtn=dtn_info,
        lambda1_compact=float(vals_C[1]), eigs_compact=vals_C.tolist(),
        liouville=dict(iterations=int(liou["iterations"]), residual=float(liou["residual"]),
                       note=liou.get("note", ""), area_compact=area_g, area_compact_exact=area_g_exact,
                       chi_discrete=float(chi_discrete),
                       eps_central=eps_central,          # sup|u| on complement of height-1 horoballs
                       eps_outside=eps_outside,          # sup|u| off all length-ell horoballs (short cusps fully inside)
                       eps_long=eps_long,                # same, cusps with k < ell cut out at height 1
                       max_abs_u_thick=float(np.abs(phi[mesh.thick_nodes]).max()),
                       min_u=float(phi_b.min()), max_u=float(phi_b.max()),
                       u_per_cusp=u_cusp),
        weyl=weyl,
        graph=gf,
        delta_compact_minus_cusped=(None if lam_c is None else float(vals_C[1] - lam_c)),
        delta_compact_minus_thick=float(vals_C[1] - vals_T[1]),
        wallclock_s=time.time() - T0,
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        log("wrote %s" % args.out)
    else:
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
