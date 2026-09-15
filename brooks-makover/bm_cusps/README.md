# Cusps und Kompaktifizierung: Brooks-Makover-Flächen numerisch

Code zu **Punkt 3** (Schrohe-Linie): Vergleich der Spektrallücke einer zufälligen
Brooks-Makover-Fläche `S` (2n ideale Dreiecke, V Cusps) mit der ihrer
Poincaré-Koebe-Kompaktifizierung `S̄` (Geschlecht g = n/2 + 1 − V/2), plus der
konforme Faktor φ zwischen beiden Metriken als Funktion der Cusp-Längen.

## Was gerechnet wird

Pro Stichprobe (`n`, `seed`):

| Größe | Bedeutung |
|---|---|
| `V`, `cusp_lengths`, `genus` | Kombinatorik des zufälligen kubischen Ribbon-Graphen (Konfigurationsmodell) |
| `lambda1_thick` | λ₁ des dicken Teils `S_Y` (Cusps an Horozyklen der Länge `L0` abgeschnitten, Neumann) |
| `lambda1_cusped` | kleinster **L²-Eigenwert von S unterhalb 1/4** über die exakte Cusp-DtN-Bedingung (Moden `y^{1/2−ν}`, `√y K_ν`), oder `null`, wenn keiner existiert |
| `lambda1_compact` | λ₁ der **uniformisierten** Kompaktifizierung: Liouville-Gleichung `Δ₀φ = K₀ + e^{2φ}` (Newton), dann `K u = λ M(ρ₀ e^{2φ}) u` |
| `liouville.phi_per_cusp` | Mittel von φ auf dem Höhe-1-Horozyklus jedes Cusps — die Größe, die im Theorem als ε(ℓ_min) auftaucht |
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
PSL(2,ℤ) (ℓ = 2 arccosh(tr(w)/2), vollständig unterhalb ℓ_cut = 2 arccosh((W+2)/2)).
`graph_scan.py` screent 10⁵–10⁷ Seeds und exportiert Extremfälle als Parameterdatei
für die FEM-Stufe; die Features landen auch in jedem `run_bm.py`-JSON (`graph`) —
neu berechnet oder mit `--graph-from scan.csv` aus Stufe 1 übernommen
(Konsistenzcheck über V und Geschlecht, Feld `graph_source`).
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
python aggregate.py            # summary.csv + Tabellen (auch Richardson-extrapoliert)
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

Beobachtung, die direkt das Theorem füttert: φ auf dem Höhe-1-Horozyklus ist
≈ −0.01 … −0.3 für lange Cusps (k ≥ 15), aber ≈ −1.3 für k = 2 und ≈ −3.8 für
k = 1 — die Kompaktifizierung ist nur außerhalb kurzer Cusps nahe an der
Cusp-Metrik. Die Konstante ε(ℓ_min) ist also real und groß für ℓ_min ~ 1.

## Bekannte Einschränkungen

* Der zentrale Bereich jedes Dreiecks wird als Polygon durch die Horozyklus-Knoten
  vermascht (Sehne statt Bogen): geometrischer Fehler O(h²), gleicher Ordnung wie
  der P1-Fehler; wird durch Richardson mit extrapoliert.
* Keine Zertifizierung der Eigenwerte (das wäre die Strohmaier–Uski-Linie aus
  Punkt 1); die Zahlen sind konvergente FEM-Näherungen.
* Modell ist Brooks–Makover (Belyi-Flächen), nicht Weil–Petersson.
* Der Weyl-Fit braucht `--neig` ≳ 5·g, um das Geschlecht verlässlich zu „hören";
  für große n ist er mit 60 Eigenwerten nur grob (P1 überschätzt hohe Eigenwerte).

## Dateien

`bmsurf.py` Kombinatorik + Netz · `bmgraph.py` Graphenstufe · `bmfem.py` Assemblierung,
DtN, Liouville, Eigenlöser · `run_bm.py` Treiber · `graph_scan.py` Screening ·
`make_params.py`, `slurm_bm.sbatch`, `slurm_scan.sbatch` Job-Arrays · `aggregate.py`
Auswertung · `EXPERIMENTS.md` Experiment-Matrix · `results_example/` Testläufe.
