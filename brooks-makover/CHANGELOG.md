# 0.4.2

- Validation twist scan now uses `vals[1]` for lambda_1; triple-cluster means are reported separately only.
- Twist refinement preserves the normalized Fenchel--Nielsen twist when changing mesh resolution.
- Bolza validation now has hard assertions for side pairings, vertex class, zero mode, area/topology proxy, and lambda_1 accuracy.
- Removed the incorrect claim that Bolza global maximality follows from the KMP upper bound; KMP is described only as a near-sharp upper bound.
- Result schema 5 records all numerical run parameters and explicit solver-convergence/analysis-eligibility flags.
- SLURM resume validates n, seed, h, L0, T_ext, neig, W, code/schema/graph versions, and compact-solver convergence.
- Quarantined files are moved outside the default result glob and are explicitly ignored by analysis/aggregation.
- Non-analysis-eligible results are excluded from downstream statistics by a mandatory quality filter.
- Updated the stale H4 selftest text assertion.

# Changelog

## 0.4.1 (2026-09-16) — Ensemble/H4/Resume hardening

- `aggregate.py` uses one independent surface per `(n, seed)` for ensemble summaries; the finest mesh is selected. `summary.csv` is deduplicated, while `summary_all_resolutions.csv` preserves every mesh for convergence/Richardson diagnostics.
- H4 now fits `c0 = F(k-bin) + A_n + beta·X` with categorical `n` fixed effects. The reported partial R² is the incremental contribution of global graph covariates after controlling for both cusp length and size. Bootstrap resampling is clustered by surface and stratified within `n`. Output explicitly states that this is locality evidence, not a proof.
- Scheibenprofil wording tightened: agreement on sampled horocycles is evidence consistent with a radial disc model; it does not identify the exact local metric.
- Added `validate_result.py`; `slurm_bm.sbatch` skips an existing result only after schema, parameters, graph-feature semantics and required blocks validate. Invalid files are quarantined and recomputed.
- Repository-root `selftest.py` is now a wrapper around the single canonical `bm_cusps/selftest.py`. New regression tests cover aggregate deduplication, SLURM result validation and removal of a purely `n`-confounded H4 signal.
- Documentation updated to version 0.4.1 and the repaired completeness cutoff `2 arccosh((W+1)/2)`.
- 3D print documentation now states explicitly that the STL/OBJ is a topological demonstrator, not a metric/FEM embedding.


## 0.3.0 (2026-09-16) — Review-Fixes, ein kanonischer Stand
* **Ein Paket, eine Version.** `version.py` (`__version__`, `SCHEMA_VERSION`); jede JSON-Ausgabe
  trägt `schema_version`/`code_version`, `analyze.py`/`aggregate.py` verweigern ältere Schemata mit
  klarer Meldung. `selftest.py` prüft in ~30 s, ob das Verzeichnis den aktuellen Stand enthält
  (Regressionstest n=16/seed=4/W=10 → 7 Geodäten, Spurformel-Check, JSONL-Scan, Schema, Scheibenformel).
  Ältere Kopien (insbesondere Root-Duplikate aus früheren Lieferungen) löschen.
* Zählfehler: `n_geodesics_below_cut` zählte alle Wort-Geodäten; jetzt nur unterhalb der Schranke.
  Schranke korrigiert auf 2 arccosh((W+1)/2) (minimale Spur eines gemischten Wortes ist w+1).
* Kappen verfeinern mit h (Kettenabstand ≤ h/2, Ringzahl ∝ Knotenzahl).
* DtN: „kein L²-Eigenwert unter 1/4" ist eine Aussage über das diskrete Problem (`certified: false`).
* Konformer Faktor heißt u (φ = Eigenfunktion). Gespeichert: Mittel und sup auf Höhe-1- und
  Länge-ℓ-Horozyklen je Cusp; `eps_h_central`, `eps_h_outside(ℓ)`, `eps_h_long(ℓ)` = sup des
  **diskreten** Faktors auf Teilmengen von S_Y (nie auf Kappen; keine Aussage über den ganzen
  ursprünglichen Cusp). Konforme Dichte c₀ je Cusp und Scheibenformel in `analyze.py h4`.

## 0.2.0 — Graphenstufe und Funnel
* `bmgraph.py` (Adjazenz/NB-Spektrum, Zyklen, Tangles, Cusps, L/R-Längenspektrum), `graph_scan.py`
  (CSV + JSONL), `run_bm.py --graph-from`, `EXPERIMENTS.md` mit H1–H6, `analyze.py` (H1 mit freiem L,
  H4 mit Cluster-Bootstrap).

## 0.1.0 — Erste Version
* Brooks-Makover-Kombinatorik, hyperbolisches P1-Netz, exakte Cusp-DtN, Liouville-Uniformisierung,
  SLURM-Array.
