# Cusps und Kompaktifizierung: Brooks-Makover-Flächen numerisch

**Version 0.4.3 — dies ist der einzige kanonische Stand.** Vor dem Start auf dem Cluster
`python selftest.py` im Arbeitsverzeichnis ausführen; der Test schlägt fehl, wenn dort ein
älterer Code-Stand liegt (Root-Duplikate früherer Lieferungen bitte löschen). Änderungen:
`CHANGELOG.md`.

Code zu **Punkt 3** (Schrohe-Linie): Vergleich der Spektrallücke einer zufälligen
Brooks-Makover-Fläche `S` (2n ideale Dreiecke, V Cusps) mit der ihrer
Poincaré-Koebe-Kompaktifizierung `S̄` (Geschlecht g = n/2 + 1 − V/2), plus der
konforme Faktor φ zwischen beiden Metriken als Funktion der Cusp-Längen.

### Result eligibility and restart safety

Version 0.4.3 writes a `run_parameters`, `quality`, and `provenance` block to every result. Compact and cusped/DtN branches have separate eligibility flags. Downstream analyses recompute eligibility from the numerical payload instead of trusting stored flags; non-finite values therefore invalidate the affected branch even if metadata claims success. SLURM restart validation checks all numerical parameters, schema/code/graph-feature provenance, current native numerical provenance, and the requested branch convergence. Rejected cached files are moved to `results/quarantine/` with a `.json.quarantine` suffix so they cannot match the normal `results/*.json` glob. See `RESULT_PROVENANCE.md` for the strict distinction between native production results and metadata-migrated archived examples.


## Was gerechnet wird

Pro Stichprobe (`n`, `seed`):

| Größe | Bedeutung |
|---|---|
| `V`, `cusp_lengths`, `genus` | Kombinatorik des zufälligen kubischen Ribbon-Graphen (Konfigurationsmodell) |
| `lambda1_thick` | λ₁ des dicken Teils `S_Y` (Cusps an Horozyklen der Länge `L0` abgeschnitten, Neumann) |
| `lambda1_cusped` | akzeptierter kleinster **L²-Eigenwert von S unterhalb 1/4** über die exakte Cusp-DtN-Bedingung. Bei `root_converged` ist er endlich; bei einem konvergierten `no_l2_detected` ist er `null`; bei fehlgeschlagener DtN-Rechnung ist er ebenfalls `null`, aber `quality.cusped_analysis_eligible=false`. Ein letzter unkonvergierter Iterationswert wird ausschließlich als `lambda1_cusped_raw` zur Diagnose gespeichert und nie statistisch verwendet. |
| `lambda1_compact` | λ₁ der **uniformisierten** Kompaktifizierung: Liouville-Gleichung `Δ₀φ = K₀ + e^{2φ}` (Newton), dann `K u = λ M(ρ₀ e^{2φ}) u` |
| `liouville.u_per_cusp`, `eps_h_*` | diskreter konformer Faktor u_h (ḡ_h = e^{2u_h} g₀): Mittel und sup auf Höhe-1- und Länge-ℓ-Horozyklen je Cusp; `eps_h_central/outside(ℓ)/long(ℓ)` = sup|u_h| auf Teilmengen von S_Y (dort g₀ = g). Vergleichskonstanten der *berechneten* Metrik auf dem genannten Gebiet, nicht zertifiziert; Kappen gehen nie ein, über den ganzen ursprünglichen Cusp gibt es keinen beidseitigen Vergleich (u → −∞ an der Punktierung) |
| `weyl` | Steigung der Zählfunktion von S̄; Weyl: Area/4π = g − 1 („Geschlecht hören") |
| Checks | diskretes Gauß-Bonnet (Σc = 2πχ, Area(S̄) = 4π(g−1)), Flächeninhalt von S_Y vs. exakt |

Numerischer Kern: in 2D ist die Dirichlet-Energie konform invariant, daher wird
jedes P1-Element mit der **euklidischen** Steifigkeit in seiner eigenen Karte
assembliert (Halbebenen-Karte je Ecke, konforme Scheibenkarte `w = e^{2πi z/k}`
für die Caps); die Metrik steckt nur in der Massenmatrix (Quadratur Grad 4).

## Graphenstufe (billig) und Funnel

`bmgraph.py` berechnet aus dem Ribbon-Graphen allein: Adjazenz- und
Non-Backtracking-Spektrum, Taillenweite, einfache Zyklen und Tangle-Proxy
(exakte Enumeration, gegen tr Bˡ verifiziert), Cusp-Statistik und das
**Längenspektrum** der gecuspten Fläche über die L/R-Cutting-Sequence-Wörter in
PSL(2,ℤ) (ℓ = 2 arccosh(tr(w)/2), vollständig unterhalb ℓ_cut = 2 arccosh((W+1)/2)).
`graph_scan.py` screent 10⁵–10⁷ Seeds und exportiert Extremfälle als Parameterdatei
für die FEM-Stufe; die Features landen auch in jedem `run_bm.py`-JSON (`graph`) —
neu berechnet oder mit `--graph-from scan_n*.jsonl` aus Stufe 1 übernommen. Scan-Schema,
Graph-Feature-Semantik und `W` werden dabei hart geprüft; alte Scans werden abgelehnt statt
stillschweigend in ein aktuelles Ergebnis übernommen. Konsistenzcheck zusätzlich über V und Geschlecht.
Experiment-Matrix und Hypothesen H1–H6: `EXPERIMENTS.md`.

## Installation und Aufruf

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, scipy, triangle (Shewchuk)

python run_bm.py --n 32 --seed 1 --h 0.1 --L0 1.0 --neig 40 --out results/bm_n32_s1_h0.1.json
```

Optionen: `--h` hyperbolische Maschenweite, `--L0` Länge des Abschneide-Horozyklus
(Cap-Anschluss), `--T-ext` Tiefe der Cusp-Verlängerung für die DtN-Bedingung (6.0),
`--neig` Zahl der Eigenwerte von S̄ für den Weyl-Fit.

## SLURM

```bash
python make_params.py --n 16 32 64 128 256 --seeds 20 --h 0.1 0.07   # zwei h → Richardson
mkdir -p logs results
sbatch --array=1-$(wc -l < params.txt)%16 slurm_bm.sbatch
python aggregate.py            # summary.csv: eine Zeile pro (n,seed); alle h separat in summary_all_resolutions.csv
```

Ressourcen (1 Kern, SciPy/SuperLU): n=32 ≈ 15 s (34k Knoten), n=128 ≈ 5 min
(150k Knoten), n=256 ≈ 15–20 min; Speicher < 8 GB bis n=256. Für n ≳ 1000
lohnt der Umstieg des Eigenlösers auf SLEPc/MUMPS (Shift-Invert), die
Assemblierung bleibt.

## Validierung (n=32, seed=1, gleiche Fläche, vier Maschenweiten)

| h | Knoten | λ₁(S_Y) | λ₁(S) | λ₁(S̄) | rel. Flächenfehler S_Y |
|---|---|---|---|---|---|
| 0.20 | 13 182 | 0.25766 | 0.214983 | 0.393391 | 3.5e-3 |
| 0.14 | 21 207 | 0.25664 | 0.213785 | 0.390574 | 1.8e-3 |
| 0.10 | 34 236 | 0.25552 | 0.213261 | 0.389379 | 8.7e-4 |
| 0.07 | 61 006 | 0.25466 | 0.212949 | 0.388647 | 4.4e-4 |

Konvergenz sauber O(h²); Richardson (0.1, 0.07): λ₁(S̄) → 0.38794, λ₁(S) → 0.21265.
Liouville-Newton konvergiert in 5–6 Schritten auf |R| ~ 1e-13, Gauß-Bonnet stimmt
bis auf den O(h²)-Geometriefehler.

Beobachtung, die direkt das Theorem füttert: u auf dem Höhe-1-Horozyklus ist
≈ −0.01 … −0.3 für lange Cusps (k ≥ 15), aber ≈ −1.3 für k = 2 und ≈ −3.8 für
k = 1. Auf Horozyklen fester *Länge* ℓ ist u in den bisherigen Tests fast k-unabhängig; die
gesampelten Profilwerte werden sehr gut durch eine Scheibenprofil-Formel mit einer effektiven
Konstante c₀ je Cusp beschrieben (EXPERIMENTS.md, H4). Ein guter Fit entlang endlich vieler
Horozyklen identifiziert **nicht** die exakte lokale Metrik und ersetzt keinen analytischen
Eindeutigkeits- oder Fehlerbeweis.
Die Kappen sind über die Kettenauflösung (Knotenabstand ≤ h/2, Ringzahl ∝ Knotenzahl)
an h gekoppelt; ihr Beitrag zu λ₁ liegt bei den getesteten h unter 10⁻⁶ relativ.

## Bekannte Einschränkungen

* Der zentrale Bereich jedes Dreiecks wird als Polygon durch die Horozyklus-Knoten
  vermascht (Sehne statt Bogen): geometrischer Fehler O(h²), gleicher Ordnung wie
  der P1-Fehler; wird durch Richardson mit extrapoliert.
* Keine Zertifizierung der Eigenwerte (das wäre die Strohmaier–Uski-Linie aus
  Punkt 1); die Zahlen sind konvergente FEM-Näherungen. Insbesondere ist
  `lambda1_cusped: null` („keiner unter 1/4") eine Aussage über das diskrete
  Problem, kein Ausschlusszertifikat.
* Die `eps_h_*` sind Suprema des P1-Faktors, keine Schranken für den exakten
  Uniformisierungsfaktor; der Abstand ist der Diskretisierungsfehler O(h²)
  (Richardson schätzt ihn, beweist ihn nicht).
* Modell ist Brooks–Makover (Belyi-Flächen), nicht Weil–Petersson.
* Der Weyl-Fit braucht `--neig` ≳ 5·g, um das Geschlecht verlässlich zu „hören";
  für große n ist er mit 60 Eigenwerten nur grob (P1 überschätzt hohe Eigenwerte).

## Dateien

`bmsurf.py` Kombinatorik + Netz · `bmgraph.py` Graphenstufe · `bmfem.py` Assemblierung,
DtN, Liouville, Eigenlöser · `run_bm.py` Treiber · `graph_scan.py` Screening ·
`make_params.py`, `slurm_bm.sbatch`, `slurm_scan.sbatch` Job-Arrays · `validate_result.py` prüft vorhandene Ergebnisse vor SLURM-Skip · `aggregate.py`
Auswertung · `analyze.py` H1-Fit (freies L) und H4-`c0`-Diagnostik mit k- und n-Fixed-Effects sowie nach n geschichtetem Cluster-Bootstrap; pro `(n,seed)` nur feinste Netzweite ·
`selftest.py` Regressionstest · `version.py` · `CHANGELOG.md` · `EXPERIMENTS.md`
Experiment-Matrix · `results_example/` Testläufe.
