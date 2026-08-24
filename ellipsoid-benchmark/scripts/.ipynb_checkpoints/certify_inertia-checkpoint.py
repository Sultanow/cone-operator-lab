#!/usr/bin/env python3
"""Certify (and, where provably necessary, correct) the merged gap-free
Hankel spectrum against exact Sylvester inertia counts, PER CLUSTER.

For symmetric K, SPD M, the LDL^T inertia of (K - lam*M) counts the
generalized eigenvalues below lam exactly. Probing the midpoints
between consecutive accepted clusters certifies every cluster
individually: the inertia difference across a cluster's bracket IS its
multiplicity. Compensating errors (a spurious level plus a missed one
in the same tile) are impossible to miss at this granularity.

Two modes, so the ~N probes parallelize over a small SLURM array:

  probes:   --task-index i --n-tasks T
            computes inertia at chunk i of the global probe list,
            writes inertia_probes_{i:03d}.csv (idempotent: skips if
            the file exists and is complete).
  assemble: loads all probe files, certifies each cluster, localizes
            certified-but-unclaimed levels by bisection (cheap, done
            inline), and writes
              full_spectrum_eigs_certified.txt
              certification_report.csv
              certification_summary.json

Every change is justified by inertia counts alone; no reference
spectrum is consulted. Requires pymumps (MUMPS LDL^T with inertia).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from mumps import DMumpsContext
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pymumps not available. Install with system/module MUMPS:\n"
        "  pip install --user pymumps   (needs libmumps-seq)\n"
        "or: module avail mumps petsc; or conda/mamba install mumps-seq, "
        "then pip install pymumps") from exc

from hearing_ellipsoid_bench.fem.assembly import load_or_create_problem
from hearing_ellipsoid_bench.geometry.ellipsoid import (
    true_ellipsoid_volume,
    ellipsoid_surface_area,
    ellipsoid_integrated_mean_curvature,
)
from hearing_ellipsoid_bench.solvers.fdm_block import weyl_density


class InertiaOracle:
    """Exact counting oracle N(lam); MUMPS analysis reused per values."""

    def __init__(self, K, M, verbose=True):
        Ku = sp.triu(K, format="csr")
        Mu = sp.triu(M, format="csr")
        K_al = (Ku + 0.0 * Mu).sorted_indices()
        M_al = (Mu + 0.0 * Ku).sorted_indices()
        assert np.array_equal(K_al.indices, M_al.indices)
        assert np.array_equal(K_al.indptr, M_al.indptr)
        self.kvals = K_al.data.copy()
        self.mvals = M_al.data.copy()
        coo = K_al.tocoo()
        self.rows = (coo.row + 1).astype(np.int32)
        self.cols = (coo.col + 1).astype(np.int32)
        self.n = K.shape[0]
        self.verbose = verbose
        self.cache = {}
        self.n_factor = 0
        self.ctx = DMumpsContext(sym=2, par=1)
        self.ctx.set_silent()
        self.ctx.id.n = self.n
        self.ctx.set_centralized_assembled(self.rows, self.cols, self.kvals)
        self.ctx.id.icntl[13] = 40      # ICNTL(14): workspace relaxation %
        t0 = time.perf_counter()
        self.ctx.run(job=1)
        if verbose:
            print(f"MUMPS analysis: {time.perf_counter() - t0:.1f} s",
                  flush=True)

    def count_below(self, lam):
        key = float(lam)
        if key in self.cache:
            return self.cache[key]
        vals = self.kvals - lam * self.mvals
        self.ctx.set_centralized_assembled_values(vals)
        t0 = time.perf_counter()
        self.ctx.run(job=2)
        neg = int(self.ctx.id.infog[11])   # INFOG(12): negative pivots
        self.n_factor += 1
        if self.verbose:
            print(f"  inertia(lam={lam:.9f}) = {neg}   "
                  f"[{time.perf_counter() - t0:.1f} s, #{self.n_factor}]",
                  flush=True)
        self.cache[key] = neg
        return neg

    def close(self):
        try:
            self.ctx.destroy()
        except Exception:
            pass


def load_clusters(root):
    acc = pd.read_csv(root / "full_spectrum_accepted.csv")
    lam_col = next(c for c in acc.columns if c.lower() in
                   ("lambda", "lam", "eigenvalue"))
    mult_col = next(c for c in acc.columns if "mult" in c.lower())
    cl = acc[[lam_col, mult_col]].sort_values(lam_col).to_numpy()
    return cl[:, 0].astype(float), cl[:, 1].astype(int)


def probe_points(lams, lam_top):
    """Global probe list: 0, all midpoints between consecutive distinct
    cluster positions, and lam_top (> last cluster)."""
    mids = 0.5 * (lams[1:] + lams[:-1])
    top = max(lam_top, lams[-1] * (1 + 1e-9) + 1e-12)
    return np.concatenate([[0.0], mids, [top]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["probes", "assemble"], required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--b", type=float, default=1.5)
    p.add_argument("--c", type=float, default=2.3)
    p.add_argument("--mesh-h", type=float, default=0.06)
    p.add_argument("--order", type=int, default=2)
    p.add_argument("--lambda-top", type=float, default=None,
                   help="upper certification bound; default: last "
                        "cluster * (1 + 1e-9)")
    p.add_argument("--task-index", type=int, default=0)
    p.add_argument("--n-tasks", type=int, default=1)
    p.add_argument("--bracket-frac", type=float, default=0.05)
    args = p.parse_args()

    lams, mults = load_clusters(args.root)
    lam_top = (args.lambda_top if args.lambda_top is not None
               else float(lams[-1]) * (1 + 1e-9))
    probes = probe_points(lams, lam_top)

    if args.mode == "probes":
        chunks = np.array_split(np.arange(len(probes)), args.n_tasks)
        idx = chunks[args.task_index]
        out = args.root / f"inertia_probes_{args.task_index:03d}.csv"
        if out.exists():
            done = pd.read_csv(out)
            if len(done) == len(idx):
                print(f"{out} complete; nothing to do.")
                return
        print(f"task {args.task_index}/{args.n_tasks}: "
              f"{len(idx)} probes", flush=True)
        _, K, M, _ = load_or_create_problem(
            args.data_root, a=args.a, b=args.b, c=args.c,
            mesh_size=args.mesh_h, order=args.order, force_remesh=False)
        oracle = InertiaOracle(K, M)
        rows = []
        for k, i in enumerate(idx):
            rows.append({"probe_index": int(i), "lam": float(probes[i]),
                         "count_below": oracle.count_below(probes[i])})
            if (k + 1) % 10 == 0 or k + 1 == len(idx):
                pd.DataFrame(rows).to_csv(out, index=False)  # checkpoint
        oracle.close()
        print(f"wrote {out}")
        return

    # ---- assemble ----
    parts = sorted(args.root.glob("inertia_probes_*.csv"))
    if not parts:
        raise SystemExit("no inertia_probes_*.csv found; run probes first")
    pr = pd.concat([pd.read_csv(f) for f in parts]).drop_duplicates(
        "probe_index").sort_values("probe_index")
    if len(pr) != len(probes):
        missing = sorted(set(range(len(probes)))
                         - set(pr["probe_index"].astype(int)))
        raise SystemExit(f"{len(missing)} probes missing, e.g. "
                         f"{missing[:10]}; run remaining probe tasks")
    counts = pr["count_below"].to_numpy()

    V = true_ellipsoid_volume(args.a, args.b, args.c)
    S = ellipsoid_surface_area(args.a, args.b, args.c)
    C = ellipsoid_integrated_mean_curvature(args.a, args.b, args.c)

    def spacing_fn(lam):
        return 1.0 / max(weyl_density(lam, V, S, C), 1e-12)

    # per-cluster certification: bracket (probe[i], probe[i+1]] holds
    # cluster i
    actions = []
    surplus_intervals = []
    for i, (lam, mult) in enumerate(zip(lams, mults)):
        cert = int(counts[i + 1] - counts[i])
        if cert == mult:
            actions.append({"action": "confirm", "lambda": lam,
                            "multiplicity": int(mult),
                            "bracket_a": probes[i],
                            "bracket_b": probes[i + 1]})
        elif cert == 0:
            actions.append({"action": "remove", "lambda": lam,
                            "multiplicity": int(mult), "certified": 0,
                            "bracket_a": probes[i],
                            "bracket_b": probes[i + 1]})
        else:
            actions.append({"action": "adjust", "lambda": lam,
                            "multiplicity": int(mult),
                            "certified": cert,
                            "bracket_a": probes[i],
                            "bracket_b": probes[i + 1]})
            if cert > mult:
                # the bracket certifiably holds cert levels whose
                # positions are only partly known; bisection below
                # re-localizes ALL of them (the claimed cluster is
                # dropped from the output in favour of the brackets)
                surplus_intervals.append(
                    (probes[i], probes[i + 1], cert, i))

    n_below0 = int(counts[0])
    if n_below0:
        print(f"WARNING: {n_below0} certified levels below lambda=0 ?!")

    # localize certified-but-unclaimed levels (needs a few extra
    # factorizations; done inline, sequential and cheap)
    added = []
    if surplus_intervals:
        print(f"localizing {len(surplus_intervals)} surplus intervals "
              f"by bisection ...", flush=True)
        _, K, M, _ = load_or_create_problem(
            args.data_root, a=args.a, b=args.b, c=args.c,
            mesh_size=args.mesh_h, order=args.order, force_remesh=False)
        oracle = InertiaOracle(K, M)
        for lo_l, hi_l, k in [(a_, b_, c_) for a_, b_, c_, _ in
                              surplus_intervals]:
            stack = [(lo_l, hi_l, k + 0)]
            # note: bracket also contains the claimed cluster; bisection
            # separates surplus levels from it down to bracket_frac
            while stack:
                a_i, b_i, cnt = stack.pop()
                width_target = args.bracket_frac * spacing_fn(
                    0.5 * (a_i + b_i))
                if b_i - a_i <= width_target:
                    added.append({"action": "add",
                                  "lambda": 0.5 * (a_i + b_i),
                                  "multiplicity": int(cnt),
                                  "bracket_a": a_i, "bracket_b": b_i,
                                  "uncertainty": 0.5 * (b_i - a_i)})
                    continue
                mid = 0.5 * (a_i + b_i)
                left = (oracle.count_below(mid)
                        - oracle.count_below(a_i))
                if left:
                    stack.append((a_i, mid, left))
                if cnt - left:
                    stack.append((mid, b_i, cnt - left))
        oracle.close()

    # NOTE on "adjust" with surplus: the bisection above re-localizes
    # ALL levels in that bracket, including the claimed cluster, so we
    # drop the cluster from the output and let the certified brackets
    # speak. For deficit adjustments (cert < mult) the cluster stays
    # with the certified multiplicity.
    out = []
    surplus_probe_ids = {i for *_x, i in surplus_intervals}
    for i, act in enumerate(actions):
        if act["action"] == "confirm":
            out += [act["lambda"]] * act["multiplicity"]
        elif act["action"] == "adjust":
            if i in surplus_probe_ids:
                continue  # replaced by bisection brackets
            out += [act["lambda"]] * act["certified"]
    for act in added:
        out += [act["lambda"]] * act["multiplicity"]
    out = np.sort(np.array(out))

    rep = pd.DataFrame(actions + added)
    rep.to_csv(args.root / "certification_report.csv", index=False)
    np.savetxt(args.root / "full_spectrum_eigs_certified.txt", out,
               fmt="%.17e")
    n_changes = int((rep["action"] != "confirm").sum())
    total_certified = int(counts[-1] - counts[0])
    summary = {
        "n_certified_levels_written": int(len(out)),
        "inertia_total_in_range": total_certified,
        "consistent": bool(len(out) == total_certified),
        "n_clusters_input": int(len(lams)),
        "n_actions": len(rep),
        "n_changes": n_changes,
        "actions_by_type": rep["action"].value_counts().to_dict(),
        "lambda_top": lam_top,
    }
    (args.root / "certification_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote full_spectrum_eigs_certified.txt, "
          "certification_report.csv, certification_summary.json")


if __name__ == "__main__":
    main()
