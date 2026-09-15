#!/usr/bin/env python3
"""
run_bm.py -- one Brooks-Makover sample: cusped surface S vs. its compactification S^bar.

Per sample (JSON):
  combinatorics    n, V (cusps), cusp lengths k_c, genus g, chi
  lambda1_thick    first eigenvalues of S_Y (cusps cut at horocycles of length L0, Neumann)
  lambda1_neumann  same with cusps extended by T_ext (Neumann far out in the cusps)
  lambda1_cusped   smallest L^2 eigenvalue of S below 1/4 (exact cusp DtN), or null
  lambda1_compact  first eigenvalues of the uniformized compactification S^bar
  liouville        Newton diagnostics, discrete Gauss-Bonnet, conformal factor phi per cusp
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
    """Row of a graph_scan.py CSV for (n, seed), with types restored; None if absent."""
    import csv
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if int(r["n"]) == n and int(r["seed"]) == seed:
                out = {}
                for k, v in r.items():
                    if v in ("", "None"):
                        out[k] = None
                    elif v in ("True", "False"):
                        out[k] = v == "True"
                    else:
                        try:
                            out[k] = int(v)
                        except ValueError:
                            out[k] = float(v)
                return out
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
                    help="scan CSV from graph_scan.py; reuse its row for (n, seed) instead of recomputing")
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
        gf.setdefault("simple_cycles", {l: gf.pop("c%d" % l) for l in range(1, 7) if "c%d" % l in gf})
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
    phi_cusp = []
    for ch in mesh.cap_chains:
        phi_cusp.append(dict(cusp=int(ch.cusp), k=int(ch.k),
                             phi_height1=float(phi[mesh.height1_rows[ch.cusp]].mean()),
                             phi_cap_boundary=float(phi[ch.node].mean())))
    log("Liouville: %d its, |R|=%.1e ; area(S^bar) %.6f vs 4pi(g-1)=%.6f ; sum c/2pi=%.5f vs chi=%d"
        % (liou["iterations"], liou["residual"], area_g, area_g_exact, chi_discrete, surf.chi))
    log("phi on height-1 horocycles: " + "  ".join("k=%d:%.3f" % (p["k"], p["phi_height1"]) for p in phi_cusp))
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
                       max_abs_phi_thick=float(np.abs(phi[mesh.thick_nodes]).max()),
                       min_phi=float(phi_b.min()), max_phi=float(phi_b.max()),
                       phi_per_cusp=phi_cusp),
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
