# Stable selection of the three-term Weyl fit window

## Purpose

The reverse-geometry fit uses

\[
N(\lambda)=A_0\lambda^{3/2}+A_1\lambda+A_2\lambda^{1/2},
\qquad
V=6\pi^2A_0,\quad S=-16\pi A_1,\quad C=6\pi^2A_2.
\]

A raw minimization over one window can select an accidental zero of the oscillatory
counting-function remainder. The paper therefore selects a stable plateau rather than
the smallest pointwise error.

## Pre-specified loss

For a center window \((s,e)\), define the nine-point neighborhood

\[
U(s,e)=\{s-25,s,s+25\}\times\{e-50,e,e+50\}.
\]

Every neighboring window must contain at least 400 eigenvalues. Consequently, only
center windows with width at least 475 and a complete neighborhood inside the common
spectrum are admissible.

For method \(m\), let \(r_{V,m}\), \(r_{S,m}\), and \(r_{C,m}\) be the relative errors
of the recovered volume, surface area, and integrated mean curvature. Define

\[
E_m(s,e)=\sqrt{\frac{r_{V,m}^2+r_{S,m}^2+r_{C,m}^2}{3}}.
\]

The stable geometry and curvature losses are

\[
L_{\mathrm{geom}}(s,e)=
\max_m\left[
\operatorname{median}_{(s',e')\in U(s,e)} E_m(s',e')
+\operatorname{IQR}_{(s',e')\in U(s,e)}r_{C,m}(s',e')
\right],
\]

\[
L_{A_2}(s,e)=
\max_m\left[
\operatorname{median}_{(s',e')\in U(s,e)}|r_{C,m}(s',e')|
+\operatorname{IQR}_{(s',e')\in U(s,e)}r_{C,m}(s',e')
\right].
\]

The IQR coefficient is fixed at \(\beta=1\). The maximum over methods makes the
selection minimax: a window cannot win because it is favorable for only one solver.
The geometry objective weights the three relative geometric errors equally. The
\(A_2\) objective is appropriate for the curvature-focused validation notebook.

## Data and exhaustive search

The calculation uses the three triaxial spectra listed in the paper's Data Availability
table and committed below `data/ellipsoid-benchmark/outputs`:

- FEM-ARPACK: 2000 eigenvalues;
- SSG: 2000 eigenvalues;
- inertia-certified block-Hankel: 2034 eigenvalues.

All spectra are first restricted to the common spectral cutoff
\(\lambda\le443.752487862109\). This leaves 1993 FEM-ARPACK values, 2000 SSG values,
and 1993 block-Hankel values. Common window indices therefore stop at 1993.

The implementation evaluates all 1,271,215 integer windows of width at least 400 for
each method. Of these, 1,043,290 center windows have a complete nine-point
neighborhood. Prefix sums of \(X^TX\) and \(X^Ty\) reduce every window fit to one
scaled 3 by 3 linear solve. The complete three-method search takes approximately 2.5
seconds on a laptop and does not require a cluster.

## Results

The stable common optima are:

| Objective | `FIT_START` | `FIT_END_MAX` | Minimax plateau loss |
|---|---:|---:|---:|
| Equal-weight geometry \((V,S,C)\) | 32 | 720 | 0.2490293445 |
| Curvature \((A_2/C)\) | 31 | 524 | 0.3370356189 |

The one-percent near-optimal set contains one window for the geometry objective
(`32,720`) and two windows for the curvature objective, spanning
`FIT_START=31...33` and `FIT_END_MAX=522...524`.

At the common geometry optimum, the center-window relative errors are:

| Method | \(r_V\) | \(r_S\) | \(r_C\) | Geometry RMS | Scaled design condition |
|---|---:|---:|---:|---:|---:|
| FEM-ARPACK | -0.010625 | -0.060588 | -0.242175 | 0.144260 | 138.38 |
| SSG | 0.000186 | 0.010146 | 0.310099 | 0.179132 | 138.58 |
| Block-Hankel | -0.010648 | -0.060795 | -0.244208 | 0.145427 | 138.38 |

At the common curvature optimum, the center-window relative errors are:

| Method | \(r_V\) | \(r_S\) | \(r_C\) | Geometry RMS | Scaled design condition |
|---|---:|---:|---:|---:|---:|
| FEM-ARPACK | -0.008019 | -0.037873 | -0.026734 | 0.027162 | 151.24 |
| SSG | -0.000297 | 0.007761 | 0.302401 | 0.174649 | 151.40 |
| Block-Hankel | -0.008046 | -0.038110 | -0.029016 | 0.028042 | 151.24 |

The method-specific stable optima are different:

| Method | Geometry optimum | Curvature optimum |
|---|---:|---:|
| FEM-ARPACK | `[30,523]` | `[30,523]` |
| SSG | `[122,937]` | `[122,937]` |
| Block-Hankel | `[30,523]` | `[30,523]` |

This separation is substantive. FEM-ARPACK and block-Hankel use the same finite-element
pencil and consequently have nearly identical window behavior, whereas SSG discretizes
the continuum problem independently. There is therefore no method-independent
pointwise optimum. The common window is a pre-defined minimax compromise; the
method-specific windows diagnose solver-dependent spectral bias.

For the curvature-focused notebook, the selected paper setting is
`FIT_START=31`, `FIT_END_MAX=524`. If a figure is intended to summarize all three
geometric coefficients with equal weight, use `FIT_START=32`, `FIT_END_MAX=720`.

## Reproduction

From `ellipsoid-benchmark`:

```bash
python -m pip install -e .
python scripts/optimize_reverse_geometry_window.py
```

The script discovers the committed repository data automatically and falls back to
`~/data` on the cluster. It writes a JSON summary and CSV audit tables to
`data/ellipsoid-benchmark/outputs/reverse_geometry_a2` when run in the repository.
Use `--dry-run` to print the complete result without writing files.

## Paper-ready methods paragraph

> We selected the index window for the three-term Weyl regression by exhaustive stable
> minimization rather than by pointwise error minimization. For every integer window
> containing at least 400 eigenvalues, we evaluated a nine-point neighborhood obtained
> by perturbing the lower index by 25 and the upper index by 50. For each eigensolver,
> the plateau loss was the neighborhood median reconstruction error plus the
> interquartile range of the signed curvature error; the common loss was the maximum
> over FEM-ARPACK, SSG, and the inertia-certified block-Hankel spectrum. All spectra
> were restricted to the common cutoff \(\lambda=443.752487862109\). Among 1,043,290
> admissible center windows, the curvature-focused minimax loss selected
> \([31,524]\), while the equal-weight \((V,S,C)\) loss selected \([32,720]\). This
> neighborhood criterion avoids selecting an isolated cancellation of the oscillatory
> counting-function remainder.

The word "optimal" in the paper should always refer to the specified loss, admissible
set, and neighborhood. The result is not an estimator-independent universal optimum.
