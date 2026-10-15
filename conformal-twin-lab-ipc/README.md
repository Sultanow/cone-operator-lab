# conformal-twin-lab (IPC update)

This copy lives in `~/conformal-twin-lab-ipc` so that the running L-BFGS sweep in
`~/conformal-twin-lab` keeps its code unchanged. Same conda environment, separate run
directory (`$CT_OUT/sweep_ipc`).

Hard-work package for the genus-three paper ("There and back again"): can Mizhaev's
polyhedron P be smoothed into an **embedded, C4-symmetric surface in R^3 that is
conformal to its circle-packing hyperbolic twin (S, h_cp)**, and how close to P can it stay?

## Work packages

| WP | Question | Status |
|----|----------|--------|
| A  | Are the 24 circle-packing radii correct? | **Proven.** Krawczyk test in Arb ball arithmetic: unique solution in a box of radius 1e-60 (log radii), at 256 and 512 bits. `src/certify_radii.py`, `slurm/certify.sbatch`. |
| B  | Does a discretely conformal, symmetric, embedded surface near P exist? | Pipeline ready, sweep prepared (this package). |
| C  | How far is P's own conformal structure from h_cp? | Next step (discrete uniformisation of P). |

## Method (WP B)

* `src/fine_mesh.py` refines every hyperbolic triangle of (S, h_cp) into n^2 geodesic
  triangles (n = 2^level, Klein-linear subdivision on the hyperboloid). Exact hyperbolic
  edge lengths, the same barycentric grid on P, and the tau-permutation of fine vertices.
* Discrete conformality (Bobenko-Pinkall-Springborn 2015): Euclidean lengths l_ij of the
  surface in R^3 must satisfy `log l_ij = (u_i+u_j)/2 + log(2 sinh(L_ij/2))` for some u.
  The residual of this equation is `conf_rms` / `conf_max`.
* C4 symmetry f(tau v) = T f(v) is exact by construction (symmetrised parametrisation).
* Energy: penalty continuation in the conformal term, plus fit to P (point-to-plane),
  discrete Willmore energy, and an optional point-to-triangle barrier with an
  intersection-guarded line search (`--barrier on`).
* Multilevel: level 2 -> 3 -> 4 by exact prolongation; checkpoint after each level;
  `--resume` skips finished levels.

## Findings so far (local runs, levels 1-2)

* P itself is far from conformal to h_cp: conf_rms 0.54, conf_max 1.6.
* Without barrier: residual driven to 3e-7 (level 1) and 8e-7 (level 2), surface within
  23-27 units RMS of P (diameter 632), but self-intersecting (54 / 98 face pairs), mainly
  between nonagons 1/2 and 7/8.
* With barrier: stays embedded but stalls at conf_rms ~ 0.06 (line search jams in contact).
* Coarse vertices still move by 22-34 units from level 1 to 2: geometry not converged yet.

## IPC solver (`--solver ipc`)

The L-BFGS + barrier combination jams in contact. `src/ipc.py` adds IPC-style contact
handling (Li et al. 2020): point-triangle **and** edge-edge barrier pairs below a small
`dhat` (0.25 fine edge lengths), recomputed at every energy evaluation, and a conservative
CCD bound on every step, so the surface can slide past itself but never pass through.
The search direction is a hand-written L-BFGS two-loop recursion with Armijo backtracking
inside the CCD-safe interval.

Local result, level 1, starting from P, `--w-fit 0.05`: **embedded**, conf_rms 6.3e-4
(conf_max 1.6e-3), still decreasing at the end of the run; the old barrier solver jammed at
6.6e-2. With `--w-fit 1`: embedded, conf_rms 1.1e-2.

```bash
sbatch slurm/sweep_ipc.sbatch      # 16 tasks, levels 1 -> 2 -> 3 -> 4, partition std
python3 scripts/collect.py /home/esul01/data/outputs/conformal_twin/sweep_ipc
```

## Target h_P (`--target P`): conformal smoothing of the polyhedron

Work package C (`src/wpc/wp_c.py`, `slurm/wp_c.sbatch`) shows that the circle-packing twin h_cp
and the polyhedron's own conformal structure h_P are different points of Teichmueller space:
over 62,306 closed curves the length ratio h_P/h_cp ranges over [0.796, 1.259] (level 3), i.e.
Thurston distance >= 0.23, stable under refinement (0.252, 0.240, 0.232, 0.230).
h_cp also depends on the arbitrary diagonals chosen inside the nonagons; h_P does not.

With `--target P` the solver keeps the surface discretely conformal to P itself (residual 0 at
the start) and smooths it. Local level-1 test: embedded, residual 6.9e-5, distance to P 5.2 RMS /
16 max units (h_cp target: 50 / 229), Willmore W/4pi 4.0 (polyhedron 7.1).

```bash
sbatch slurm/wp_c.sbatch           # minutes; writes $CT_OUT/wp_c/wp_c_report.json
sbatch slurm/sweep_hp.sbatch       # 12 tasks, levels 1 -> 4
python3 scripts/collect.py /home/esul01/data/outputs/conformal_twin/sweep_hp
```

## Alexandrov-type search (`src/alex`, `slurm/alex.sbatch`)

Question: is P the only polyhedral surface in R^3 (up to congruence) with its intrinsic flat metric?
Alexandrov's uniqueness theorem covers convex polyhedra of genus 0 only.

Facts established locally: P is infinitesimally rigid (rank 66); the graph of K is generically
globally rigid in R^3 (Gortler-Healy-Thurston stress rank 20 = n-d-1); P admits no positive
semidefinite full-rank stress, so Connelly's universal-rigidity certificate does not apply; P is
non-generic (integer, symmetric, 64 collinear vertex triples inside the nonagons).

The search solves |X_i - X_j|^2 = D_ij (exact integers) by Levenberg-Marquardt from random,
noisy and 'popped' starts, for K (odd tasks) and for nonagons folded along random other diagonals
(even tasks), and classifies solutions modulo congruence and the symmetry relabelling.

```bash
sbatch slurm/alex.sbatch                         # 128 tasks x 50,000 starts (about 40 min each)
CT_ALEX_N=200000 sbatch slurm/alex.sbatch        # larger round
python3 scripts/collect_alex.py /home/esul01/data/outputs/conformal_twin/alexandrov
```

## Complete classification of C4-symmetric realisations (`src/symm`, `slurm/symm.sbatch`)

With the symmetry X[tau v] = T X[v] the realisation problem reduces to 17 unknowns (six orbit
representatives, rotation about the axis fixed) and 22 quadratic equations. The 22 equations are
randomised to 17 and solved by projective total-degree homotopy continuation over all 2^17 = 131,072
start points; every isolated nonsingular complex solution is reached with probability one. Three runs
with independent gamma must give the same solution set. Paths ending at singular points at infinity
are irrelevant; singular finite end points (which would include flexible realisations) are reported.
The tracker was validated on random dense quadratic systems (all 2^n solutions found).

```bash
sbatch slurm/symm.sbatch                         # 384 tasks of about 40 s
python3 src/symm/collect_symm.py /home/esul01/data/outputs/conformal_twin/symmetric
```

## Visualisation (`visual/`, `scripts/export_obj.py`)

`visual/mizhaev_visual.wl` (Wolfram Language) shows the polyhedron, a conformal surface from the cluster and
the canonical twin h_P in the Poincare disk as rotatable graphics and exports the current view (PNG, PDF,
turntable GIF). The data files next to it are produced by `scripts/export_obj.py`; the OBJ files also open in
Blender, MeshLab or ParaView.

```bash
python3 scripts/export_obj.py polyhedron visual/          # P (8 nonagons) and K (56 triangles)
python3 scripts/export_obj.py RESULT_L4.npz visual/       # any surface from an embedding sweep
```

Figures of the paper from the repository:

```bash
python3 visual/fig_hp_disk.py visual/                     # Fig. 5 (canonical twin) and visual/hp_disk.json
python3 visual/fig_views.py WEAK.json STRONG.json         # Fig. 7 (three views of one object)
python3 src/wpc/roundtrip.py OUTDIR                       # Table 3 (exact way back), also run by slurm/wp_c.sbatch
```

Job status at a glance: `bash scripts/ctstat.sh [JOBID ...]`.

## Reproduce every number and figure of the paper

```bash
python3 src/experiment.py                          # src/results.json: triangulation K, circle packing h_cp
python3 src/develop.py                             # src/development.json: K unfolded in the Poincare disk
python3 src/symmetry_period_certificate.py OUT/    # homology character (6,0,-2,0) of the symmetry
python3 src/ribbon.py OUT/                         # Appendix B: ribbon graph (rotation system, checks)
sbatch slurm/certify.sbatch                        # Theorem 3: certified radii
sbatch slurm/wp_c.sbatch                           # Proposition 6, Table 3, length data for Fig. 6
python3 src/pants/cuffs.py OUT/                    # Table 4: pants decomposition and cuff lengths
sbatch slurm/alex.sbatch                           # random search for other realisations
sbatch slurm/symm.sbatch                           # Proposition 8: homotopy runs 0-2 (then --array=385-640%96 for 3-4)
python3 visual/fig_poly.py FIGS/                   # Fig. 1
python3 visual/fig_slices.py FIGS/                 # Fig. 2 (three handles)
python3 visual/fig_flowers.py FIGS/                # Fig. 3
python3 visual/fig_disk.py FIGS/                   # Fig. 4
python3 visual/fig_hp_disk.py FIGS/                # Fig. 5 (and hp_disk.json for Mathematica)
python3 visual/fig_lengths.py $CT_OUT/wp_c/length_compare.npz FIGS/   # Fig. 6
python3 visual/fig_views.py WEAK.json STRONG.json  # Fig. 7 (surfaces exported with scripts/export_obj.py)
sbatch slurm/ribbon.sbatch                         # ribbon-graph figure: 64 layout searches
python3 scripts/pick_ribbon.py $CT_OUT/ribbon      # picks the best layout -> $CT_OUT/ribbon/best/fig_ribbon.pdf
```

The first two steps reproduce the shipped `src/results.json` and `src/development.json` exactly.

## Cluster workflow

```bash
# once, on the login node (edit slurm/config.sh first if paths differ)
bash slurm/setup_env.sh
python3 scripts/make_grid.py > grid.txt              # 96 tasks
python3 scripts/make_grid.py --smoke > grid_smoke.txt

sbatch slurm/certify.sbatch                          # WP A, minutes
sbatch slurm/smoke.sbatch                            # 2 short tasks: validates env + paths
# inspect $CT_OUT/smoke/*.log, then:
sbatch slurm/sweep.sbatch                            # array 1-96, at most 32 at once
python3 scripts/collect.py /home/esul01/data/outputs/conformal_twin/sweep1
```

Partition lines are commented out (`##SBATCH --partition=CHANGE_ME`); uncomment and set
if the cluster requires one. All jobs are CPU-only (JAX on CPU, float64).

## Reading the results

`collect.py` writes `summary.csv` and prints, for the deepest level: embedded runs ranked by
conformal residual, the best immersed runs, and the Pareto front (conformal residual vs.
distance to P) among embedded runs. A **positive result** would be: embedded at every
level, conf_rms < 1e-6 at level 4, and `move` (motion of coarse vertices under refinement)
shrinking from level 3 to 4. A **negative result** (no embedded run below, say, 1e-3) is
also informative and goes into the paper as a measured obstruction near P.

## Caveats

* Discrete conformality on a fixed mesh is numerical evidence, not a proof of a smooth
  conformal embedding; convergence under refinement is the evidence to look for.
* The barrier + L-BFGS combination jams in contact. If the sweep finds no embedded
  low-residual run, the next step is a proper IPC-style Newton solver with continuous
  collision detection.
