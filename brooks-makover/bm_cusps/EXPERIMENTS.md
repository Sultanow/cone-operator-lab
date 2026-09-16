# Experiment-Matrix: Endliche Brooks-Makover-Flächen

Rahmen: Shen–Wu (arXiv:2511.02517) beweisen λ₁(S̄) > 1/4 − n^{−1/221} w.h.p. für die
kompaktifizierte Fläche, Hide–Magee (Ann. Math. 2023) „keine neuen Eigenwerte unter
1/4 − ε" für die gecuspte. Beides sagt nichts über endliche n, Fluktuationen,
Extremwerte, den Mechanismus (Cusps → Kompaktifizierung) oder die Spektralstatistik.
Genau das ist der Gegenstand.

## Pipeline (Funnel)

```
graph_scan.py  ──►  10^5–10^7 Ribbon-Graphen, ~0.07 s/Graph bei n=64      (Stufe 1)
      │               Features: Zyklen, Tangles, Cusps, μ₂, ρ₂(NB), Längenspektrum
      ▼  --select <feature> --top/--bottom
run_bm.py      ──►  10^2–10^3 Flächen, FEM: λ₁(S) via Cusp-DtN, Liouville → S̄   (Stufe 2)
      ▼
aggregate.py   ──►  summary.csv, Richardson, Korrelationen                       (Stufe 3)
```

Der Seed bestimmt den Graphen deterministisch (gleicher RNG in beiden Stufen), die
Auswahl in Stufe 1 ist also exakt reproduzierbar in Stufe 2. Stufe 1 schreibt zwei
Dateien: `scan_n*.csv` (flach, für Ranking/pandas) und `scan_n*.jsonl` (das komplette
`graph_features()`-Dictionary je Seed, verlustfreie Rohdatenquelle). Jeder JSONL-Datensatz
trägt Scan-Schema und Graph-Feature-Version. `run_bm.py --graph-from` akzeptiert nur exakt
die aktuelle Semantik und ein passendes `W`; alte Scans müssen neu erzeugt werden. So kann
ein reparierter Zählfehler nicht über historische Stage-1-Daten wieder eingeschleust werden.

## Gespeicherte Größen pro Sample

| Nr. | Größe | Quelle | Kosten |
|---|---|---|---|
| 1 | V, Cusp-Längen k_c, g, Anteil k_max/6n, #Cusps mit k ≤ 2 | Kombinatorik | ~0 |
| 2 | Taillenweite, einfache Zyklen c₁…c_W, Tangle-Proxy (nicht-einfache geschlossene NB-Wege ≤ W) | Enumeration, exakt (gegen tr B^ℓ verifiziert) | ms |
| 3 | μ₂(G), 3 − μ₂, μ_min, Ramanujan-Flag; NB-Spektrum ρ₂ | dense/sparse eig | ms–s |
| 4 | Längenspektrum von S unterhalb ℓ_cut = 2 arccosh((W+1)/2) über L/R-Wörter: Systole, #Geodäten < 2, < 3, Worttypen (#L, #R) | Kombinatorik, exakt | ms |
| 5 | λ₁(S_Y) (dicker Teil, Neumann) | FEM | s–min |
| 6 | λ₁(S): kleinster L²-Eigenwert < 1/4, exakte Cusp-DtN-Bedingung; „keiner" ist eine numerische Aussage der Diskretisierung, kein Zertifikat (Feld `certified: false`) | FEM + Wurzelsuche | s–min |
| 7 | λ₁…λ_neig(S̄) nach Liouville-Uniformisierung, Gauß-Bonnet-Check | FEM | s–min |
| 8 | u_h je Cusp: Mittel/sup auf Höhe-1- und Länge-ℓ-Horozyklen, c₀ (konforme Dichte an der Punktierung); ε_h,central, ε_h,outside(ℓ), ε_h,long(ℓ) je Fläche (diskret, nur auf S_Y) | Liouville | inklusive |
| 9 | Weyl-Steigung (Geschlecht hören) | Fit | inklusive |
| 10 | h und Richardson-Extrapolation (0.1, 0.07) auf einer Teilmenge | 2 Läufe | ×2 |

## Plan für n und Stichproben

| n | 2n Dreiecke | ⟨g⟩ ≈ n/2 | Knoten (h=0.1) | Zeit/Sample | Samples FEM | Samples Scan |
|---|---|---|---|---|---|---|
| 16 | 32 | 7 | 17k | 5 s | 500 | 10⁶ |
| 32 | 64 | 15 | 34k | 15 s | 500 | 10⁶ |
| 64 | 128 | 30 | 70k | 1 min | 300 | 10⁶ |
| 128 | 256 | 62 | 150k | 5 min | 200 | 10⁵ |
| 256 | 512 | 126 | 300k | ~20 min | 100 | 10⁵ |
| 512 | 1024 | 254 | 600k | ~1 h (SLEPc ab hier sinnvoll) | 30 | 10⁴ |

Grob 200 CPU-Stunden für die FEM-Stufe, gut im SLURM-Array parallelisierbar.
Für Paper 3 (Spektralstatistik) zusätzlich P2-Elemente oder h ≈ 0.03 auf 20–50
Flächen mit 10³–10⁴ Eigenwerten.

## Hypothesen

**H1 (Finite-Size-Gesetz).** m(n) := median(λ₁(S̄)) wird mit *freiem* Grenzwert L
gefittet: m(n) = L + C·n^{−α} bzw. m(n) = L + C/log n; erst danach der Test
H₀: L = 1/4 (z-Score aus dem Cluster-Bootstrap über Flächen je n). 1/4 wird nie in
den Fit eingebaut – ein Ergebnis wie L = 0.2517 ± 0.0041 ist beweiskräftiger als
ein erzwungenes L. Falsifiziert, wenn keine der beiden Familien die Mediane
beschreibt (RMS-Residuum ≫ Bootstrap-Streuung) oder L signifikant ≠ 1/4 (dann ist
das selbst das Resultat). Erste Punkte liegen oberhalb 1/4 (n=32: 0.39, n=128:
0.31); nur S̄ ist das richtige Objekt, für gecusptes S ist λ₁ ≤ 1/4 trivial.
Werkzeug: `python analyze.py h1`.

**H2 (Fluktuationen).** Die Verteilung von (λ₁(S̄) − median)/IQR ist bei festem n
linksschief (Tracy-Widom-artig, Analogie zu Huang–McKenzie–Yau für μ₂ regulärer
Graphen), nicht gaußsch. Test: Schiefe, Kurtosis, KS-Abstand zu TW₁ und zu N(0,1).
Nebenfrage: Anteil P(λ₁(S̄) > 1/4) als Funktion von n (Analogon zur 69 %-Frage).

**H3 (Graph → Fläche).** λ₁(S̄) − 1/4 wird durch kurze Zyklen, Tangle-Proxy und
Anzahl kurzer Cusps besser vorhergesagt als durch 3 − μ₂(G). Test: Regression auf
Trainings-n, Vorhersage auf ausgelassenem n; partielle Korrelationen. Begründung:
Friedman-Tangles und die Friedman-Ramanujan-Funktionen von Anantharaman–Monk
identifizieren Tangles, nicht die Graphenlücke, als Mechanismus der Abweichung.
Falsifiziert, wenn μ₂ allein die gleiche Vorhersagegüte erreicht.

**H4 (Kompaktifizierungs-Mechanismus, Lokalität).** Notation: u ist der konforme
Faktor, ḡ = e^{2u} g₀ (φ bleibt Eigenfunktionen vorbehalten). Gespeichert wird pro
Cusp nicht nur der Mittelwert, sondern Mittel *und* sup|u| auf dem Höhe-1-Horozyklus
sowie auf den Horozyklen fester **Länge** ℓ ∈ {1,2,4,…,64} (Höhe k_c/ℓ, tatsächliche
Länge mitgespeichert), und pro Fläche die diskreten Vergleichskonstanten ε_h,central = sup|u_h|
auf den zentralen Bereichen (S_Y ohne Höhe-1-Horobälle), ε_h,outside(ℓ) (S_Y ohne
Länge-ℓ-Horobälle; kurze Cusps k_c < ℓ tragen ihren Streifen bis zum L0-Horozyklus bei)
und ε_h,long(ℓ) (kurze Cusps bei Höhe 1 herausgeschnitten). Alle Gebiete liegen in S_Y,
wo g₀ = g ist – Kappen gehen nie ein, denn dort bezieht sich u_h auf die flache
Hilfsmetrik, und über den ganzen ursprünglichen Cusp gibt es keinen beidseitigen
Vergleich (ḡ ist an der Punktierung glatt, u → −∞ relativ zu g). Für P1 wird das
Supremum an Knoten angenommen; ε_h ist damit die Konstante in e^{−2ε_h} g ≤ ḡ_h ≤
e^{2ε_h} g für die *berechnete* Metrik auf dem genannten Gebiet. Für den exakten
Uniformisierungsfaktor fehlt eine abgesicherte Fehlerschranke (O(h²), Richardson
schätzt sie) – ein Mittelwert wäre nicht einmal das.

Befund aus der ersten Fläche (n=32, alle 10 Cusps, k von 1 bis 118), noch zu
erhärten: u hängt auf den **gesampelten** Horozyklen fester Länge nur schwach von k ab,
und die beobachteten Profilwerte lassen sich mit **einem effektiven Parameter c₀ pro Cusp**
sehr gut beschreiben. Mit der kanonischen Koordinate w = e^{2πi z/k} motiviert das lokale
Scheibenmodell
u(ℓ) = ½ log c₀ + log(2π/ℓ) − 2π/ℓ + O(e^{−4π/ℓ}); die zugehörige
Scheibenprofil-Formel
u(ℓ) = log((4π/ℓ)·r₀ρ/(ρ² − r₀²)), r₀ = e^{−2π/ℓ}, ρ = 2/√c₀, sagt u bei ℓ = 2, 4, 8, 16
aus dem Wert bei ℓ = 1 in diesem Test auf etwa 10⁻³ voraus. **Das identifiziert nicht die
exakte lokale Metrik:** ein Fit auf endlich vielen Horozyklen ist zunächst nur Evidenz, die
mit einem radialen hyperbolischen Scheibenprofil konsistent ist. Eine analytische Aussage
ḡ = c(w)|dw|² mit kontrolliertem Rest bzw. eine Eindeutigkeitsaussage bleibt Teil der
Theorem-Arbeit. Schwarz–Pick motiviert die obere c₀-Schranke; die quantitative untere
Schranke ist der Schrohe/Theorem-Teil.

H4 wird numerisch nun bewusst in zwei Teile zerlegt: (i) die **finite-size Frage**, ob
c₀(c) bei festem k mit wachsendem n gegen eine k-abhängige Grenzstruktur stabilisiert; und
(ii) die **Locality-Diagnostik**
c₀ = F(k-bin) + A_n + β₁·girth + β₂·gap + β₃·n_tangles + β₄·n_cusps_short.
Die kategorialen Fixed Effects A_n kontrollieren explizit unterschiedliche Oberflächengrößen;
der berichtete partielle R² misst nur den Zusatznutzen der globalen Graphgrößen nach Kontrolle
für k **und n**. β_i ≈ 0 bzw. kleines partielles R² ist Evidenz, die mit Lokalität konsistent
ist, **kein Beweis eines Locality Lemmas**. Cusps einer Fläche sind abhängig, daher wird der
Cluster-Bootstrap über ganze Flächen und zusätzlich **innerhalb jeder n-Klasse geschichtet**.
Mehrere Netzweiten derselben `(n,seed)`-Fläche sind keine unabhängigen Samples; Ensemble-
Auswertungen verwenden pro Fläche die feinste vorhandene Netzweite. Eine Richardson-
Extrapolation von c₀ wird erst eingesetzt, wenn ihre h-Asymptotik separat validiert ist. Werkzeug:
`python analyze.py h4 --plot h4_c0_vs_k.png`.

**H5 (Spektralstatistik).** Für S̄ folgen die entfalteten Abstände GOE
(Wigner-Surmise, Σ²(L), Δ₃(L) rigide). Für gecusptes S *nicht*: Γ hat endlichen
Index in PSL(2,ℤ), ist also arithmetisch und meist nicht-kongruent; das Spektrum
enthält das alte Spektrum des Kongruenzabschlusses (Poisson-Anteil) plus neuen
Teil. Test: Mischungsanteil als Funktion des Index des Kongruenzabschlusses.
Falsifiziert, wenn S̄ systematisch von GOE abweicht (dann modellabhängige
Statistik – auch interessant).

**H6 (Den Graphen hören).** Das Längenspektrum von S unterhalb ℓ_cut (nach Huber
durch das Laplace-Spektrum bestimmt) bestimmt Taillenweite, c₁…c₆ und die
Cusp-Längen ≤ W eindeutig; Umkehrung des L/R-Wörterbuchs ℓ = 2 arccosh(tr(w)/2)
ist bis auf Worttyp-Kollisionen (gleiche Spur, verschiedene Wörter) injektiv.
Test: Rekonstruktionsrate der Graphgrößen aus dem Längen-Multiset; Vergleich der
Anzahl kurzer Geodäten mit dem Poisson-Grenzwert (Budzinski–Curien–Petri). Der
Satz „Längenspektrum = Wortspuren der geschlossenen NB-Wege des Ribbon-Graphen"
ist rein kombinatorisch und der natürliche Lean-Kandidat (Theorem 2b).

## Was davon Theorem, Formalisierung, Empirie ist

| | rigoros | Lean | empirisch |
|---|---|---|---|
| Kompaktifizierungs-Lemma quantitativ (H4): c₀-Schranken, Scheibenformel | ✓ Ziel (obere Schranke via Schwarz–Pick sofort) | – | Scheibenformel auf 10⁻³ bestätigt (1 Fläche) |
| L/R-Wörterbuch Längenspektrum ↔ Ribbon-Graph (H6) | ✓ (klassisch, Cutting Sequences) | ✓ Kandidat | Verifiziert (tr B^ℓ-Check) |
| Alon-Boppana, Cheeger, Brooks-Makover-Geschlecht | bekannt | ✓ | – |
| H1, H2, H3, H5 | – | – | ✓ Kernaussagen der Paper 1–3 |

## Extreme suchen (Stufe 1 → 2)

```bash
python graph_scan.py --n 128 --seeds 0 100000 --out scan_n128.csv \
       --select gap --top 100 --bottom 100 --params params_extremes.txt
sbatch --array=1-200%16 slurm_bm.sbatch     # mit params.txt → params_extremes.txt
```

Ranking-Kandidaten: `gap` (3 − μ₂; NB-ρ₂ ist für Ramanujan-Graphen konstant √2 und
daher nur als Flag nützlich), `systole`, `n_tangle_walks`, `n_cusps_short`,
`n_geodesics_below_3`. Benchmark für kleine n: maximales λ₁ in Geschlecht 2 ist die
Bolza-Fläche (≈ 3.8388, Strohmaier–Uski).
