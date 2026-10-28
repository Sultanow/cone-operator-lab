# Hearing the Sides and Hearing the Edges

[![Axioms](https://img.shields.io/badge/Axioms-15%284%29%2C%20284%20%282026%29-orange)](https://www.mdpi.com/2075-1680/15/4/284) [![DOI](https://img.shields.io/badge/DOI-10.3390%2Faxioms15040284-blue)](https://doi.org/10.3390/axioms15040284) [![arXiv](https://img.shields.io/badge/arXiv-2511.23047-b31b1b)](https://arxiv.org/abs/2511.23047)

> **Box:** E. Sultanow, A. Hatziiliou, C. May, N. Kratzke, *Hearing the Edges: Recovering a 3D Rectangular Box from Dirichlet Eigenvalues*. Axioms 15 (2026), 284, [doi:10.3390/axioms15040284](https://doi.org/10.3390/axioms15040284). **Read it on [MDPI](https://www.mdpi.com/2075-1680/15/4/284).**
>
> **Rectangle:** E. Sultanow, A. Hatziiliou, *Hearing the Sides: Recovering a Planar Rectangle from Eigenvalues*. Preprint (2025), [arXiv:2511.23047](https://arxiv.org/abs/2511.23047).

This folder contains the notebooks, modules and data of two inverse spectral studies on domains whose Dirichlet spectrum is known in closed form: the planar rectangle and the three-dimensional rectangular box. Both serve as controlled benchmarks without eigensolver error and are the predecessors of the triaxial ellipsoid benchmark in [`../ellipsoid-benchmark`](../ellipsoid-benchmark).

## Hearing the Sides: the planar rectangle

The Dirichlet eigenvalues of $\Omega_{a,b}=(0,a)\times(0,b)$ are $\lambda_{m,n}=\pi^2(m^2/a^2+n^2/b^2)$. The paper recovers $a$ and $b$ from the ordered spectrum alone, without the mode indices $(m,n)$ and even when the lowest eigenvalues are missing:

- the area $ab$ follows from the slope of the counting function, $ab=4\pi\lim_{\lambda\to\infty}N(\lambda)/\lambda$;
- the side lengths follow from the fluctuations $F(\lambda)=N(\lambda)-N_{\text{smooth}}(\lambda)$: their length spectrum $S(L)=\left|\sum_k F(\lambda_k)e^{-iL\sqrt{\lambda_k}}\right|^2$ peaks at the bouncing-ball orbit lengths $2a$ and $2b$, and the area selects the right pair among the harmonics.

Working example: $a=1$, $b=3$, 10,000 eigenvalues after skipping the first five. The peaks at $L\approx1.9968$ and $L\approx6.0032$ give $(\hat a,\hat b)\approx(0.9984,\,3.0016)$.

| file | content |
|------|---------|
| [`mathematica/Hearing_Rectangle_Fourier.nb`](mathematica/Hearing_Rectangle_Fourier.nb) | Weyl fit and length spectrum, the working example of the paper (Listings 1–3, Figure 1) |
| [`python/notebooks/Hearing_Rectangle_Fourier.ipynb`](python/notebooks/Hearing_Rectangle_Fourier.ipynb) | the same in Python |
| [`mathematica/Hearing_Rectangle_Momente.nb`](mathematica/Hearing_Rectangle_Momente.nb), [`python/notebooks/Hearing_Rectangle_Momente.ipynb`](python/notebooks/Hearing_Rectangle_Momente.ipynb) | alternative moment estimator, not used in the paper |

## Hearing the Edges: the rectangular box

For $\Omega(a,b,c)=(0,a)\times(0,b)\times(0,c)$ the eigenvalues are $\lambda_{l,n,m}=\pi^2(l^2/a^2+n^2/b^2+m^2/c^2)$. The paper fits the truncated heat trace $Z(t)=\sum_{k\le N}e^{-\lambda_k t}$ on a short-time window $t\in[f_{\min}/\lambda_N,\,f_{\max}/\lambda_N]$, converts the heat coefficients into the Weyl coefficients $A_0, A_1, A_2$ and reads off

$$abc=6\pi^2A_0,\qquad ab+ac+bc=-8\pi A_1,\qquad a+b+c=4\pi^2A_2 .$$

By Vieta, $a$, $b$, $c$ are the roots of the cubic

$$x^3-(4\pi^2A_2)\,x^2+(-8\pi A_1)\,x-6\pi^2A_0=0 .$$

Working example: $\Omega(1,1.5,2.3)$ with 20,000 eigenvalues. With only 2000 eigenvalues the third coefficient $A_2$ is still recovered to better than one percent if the window starts at the stability minimum $f^*=t_{\min}\lambda_N\approx69$. Across aspect ratios $x=b/a$, $y=c/a$ on a $200\times200$ grid, $f^*$ forms a stability landscape; its $f^*=69$ plateau is fitted by an egg-shaped contour.

| file | content | paper |
|------|---------|-------|
| [`mathematica/Hearing_Box_GenerateEigenvalues.nb`](mathematica/Hearing_Box_GenerateEigenvalues.nb) | exact eigenvalues of $\Omega(1,1.5,2.3)$ | Listing 1 |
| [`mathematica/Hearing_Box_ReconstructWeylCoeffs_HeatEquation.nb`](mathematica/Hearing_Box_ReconstructWeylCoeffs_HeatEquation.nb) | heat-trace fit, Weyl coefficients, cubic reconstruction | Listings 2–4 |
| [`mathematica/Hearing_Box_ReconstructWeylCoeffs_HeatEquation_PlotStabilitätsminimumA2.nb`](mathematica/Hearing_Box_ReconstructWeylCoeffs_HeatEquation_PlotStabilit%C3%A4tsminimumA2.nb) | relative error of $A_2$ over $f$ for $N$ = 2000 to 20,000 → [`plots/plotA2ErrorMin.pdf`](plots/plotA2ErrorMin.pdf) | Figure 2 |
| [`mathematica/Hearing_Box_AnalyzeStabilitätsminimum.nb`](mathematica/Hearing_Box_AnalyzeStabilit%C3%A4tsminimum.nb) | stability landscape $f^*(x,y)$ as 3D surface | Figure 3 |
| [`python/notebooks/Hearing_Box_ReconstructWeylCoeffs_HeatEquation_CalcStabilitätsminimumA2-2D.ipynb`](python/notebooks/Hearing_Box_ReconstructWeylCoeffs_HeatEquation_CalcStabilit%C3%A4tsminimumA2-2D.ipynb) | $f^*=69$ plateau and egg-shaped contour fit | Figure 4 |
| [`renderings/box_a1.0_b1.5_c2.3.blend`](renderings/box_a1.0_b1.5_c2.3.blend) | Blender rendering of the working box | Figure 1 |
| other `Hearing_Box_*` notebooks, `vulcano*.ipynb` | further analyses of the stability minimum and the landscape (plateau, error curves, Riesz smoothing, surface fits) | – |
| [`python/modules/hearing_box.py`](python/modules/hearing_box.py) | `Box`, heat-trace fit, $A_2$ scans and stability-minimum search used by the Python notebooks | – |

[`doc/20260221_Hearing-a-box.pdf`](doc/20260221_Hearing-a-box.pdf) is the submitted manuscript (February 2026); the published version is on MDPI.

## Structure

```
rectangle-and-box/
├── README.md
├── mathematica/                 Hearing_Rectangle_*.nb, Hearing_Box_*.nb
├── python/
│   ├── modules/                 hearing_box.py, hearingutils.py
│   └── notebooks/               Hearing_Rectangle_*.ipynb, Hearing_Box_*.ipynb, vulcano*.ipynb
├── data/
│   ├── eigenvalues/             box_a1.0_b1.5_c2.3_N20000.txt
│   └── meshes/                  stability surfaces f*(x,y) of the box (CSV, mesh, metadata, PNG)
├── renderings/                  box_a1.0_b1.5_c2.3.blend
├── plots/                       plotA2ErrorMin.pdf
└── doc/                         submitted manuscript of the box paper
```

## Running the notebooks

The Python notebooks expect to be started from `python/notebooks`; they add the parent directory `python/` to `sys.path` and import `modules.hearing_box`. They need `numpy`, `scipy`, `pandas`, `matplotlib`, `plotly` and `scikit-learn`; `vulcano.ipynb` additionally uses `joblib` and `tqdm` and reads and writes the stability-surface CSVs in the current directory (the committed results are in `data/meshes`).

The Mathematica and Python notebooks load and export data via absolute paths of the author's machine, for example `C:/Users/sulta/git/cone-operator-lab/data/eigenvalues/...`. After cloning, replace this prefix with the location of your clone and insert `rectangle-and-box/`, e.g. `<clone>/rectangle-and-box/data/eigenvalues/box_a1.0_b1.5_c2.3_N20000.txt`.

## Paths cited in the papers

Both papers name files by their location before this folder existed. The tag [`layout-as-cited`](https://github.com/Sultanow/cone-operator-lab/tree/layout-as-cited) preserves that layout; the files now live here:

| cited path | new path | cited in |
|------------|----------|----------|
| `mathematica/Hearing_Rectangle_Fourier.nb` | `rectangle-and-box/mathematica/Hearing_Rectangle_Fourier.nb` | arXiv:2511.23047, Section 5 |
| `renderings/box_a1.0_b1.5_c2.3.blend` | `rectangle-and-box/renderings/box_a1.0_b1.5_c2.3.blend` | Axioms, Figure 1 |
| `mathematica/Hearing_Box_GenerateEigenvalues.nb` | `rectangle-and-box/mathematica/Hearing_Box_GenerateEigenvalues.nb` | Axioms, Listing 1 |
| `mathematica/Hearing_Box_AnalyzeStabilitätsminimum.nb` | `rectangle-and-box/mathematica/Hearing_Box_AnalyzeStabilitätsminimum.nb` | Axioms, Figure 3 |
| `python/notebooks/Hearing_Box_ReconstructWeylCoeffs_HeatEquation_CalcStabilitaetsminimumA2-2D.ipynb` | `rectangle-and-box/python/notebooks/Hearing_Box_ReconstructWeylCoeffs_HeatEquation_CalcStabilitätsminimumA2-2D.ipynb` | Axioms, Figure 4 |

The paper spells the last file name with `ae`; in the repository it is written with `ä`.

## Citation

```bibtex
@article{sultanow2026edges,
  title   = {Hearing the Edges: Recovering a {3D} Rectangular Box from
             Dirichlet Eigenvalues},
  author  = {Sultanow, Eldar and Hatziiliou, Andreas and May, Cornelius and
             Kratzke, Nane},
  journal = {Axioms},
  volume  = {15},
  number  = {4},
  pages   = {284},
  year    = {2026},
  doi     = {10.3390/axioms15040284},
  url     = {https://www.mdpi.com/2075-1680/15/4/284}
}

@misc{sultanow2025sides,
  title         = {Hearing the Sides: Recovering a Planar Rectangle from
                   Eigenvalues},
  author        = {Sultanow, Eldar and Hatziiliou, Andreas},
  year          = {2025},
  eprint        = {2511.23047},
  archivePrefix = {arXiv},
  primaryClass  = {math.SP},
  url           = {https://arxiv.org/abs/2511.23047}
}
```

## Acknowledgments

We thank Elmar Schrohe for suggesting the study of inverse spectral questions and for inspiring the investigation of how geometric information can be recovered from eigenvalue data.
