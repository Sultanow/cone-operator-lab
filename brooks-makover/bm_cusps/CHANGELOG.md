# Changelog

## 0.4.0 — provenance + H4 statistical correctness

- Stage-1 scans now carry `scan_schema_version`, `graph_feature_version`, and code provenance.
- `run_bm.py --graph-from` rejects missing/stale/incompatible scans and mismatched `W`; it never silently recomputes an explicitly requested scan.
- Result schema bumped to 4; graph-feature semantics versioned separately (v2 = repaired primitive-geodesic count).
- Ensemble analyses identify a random surface by `(n, seed)`; repeated mesh widths are not independent samples. The finest `h` is selected and reported.
- H4 residual regression now tests the stated puncture-density hypothesis `c0 - F(k)` rather than `u_height1_mean - F(k)`. Bootstrap clusters remain whole surfaces.
- H4 plots now show `c0` versus cusp length with the unit-disc reference `c0=4`.
- Self-test covers stale-scan rejection, resolution deduplication, and a synthetic `c0` regression recovery test.

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
