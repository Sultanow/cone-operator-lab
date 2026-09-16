"""
Step 1: level-spacing statistics for random genus-2 surfaces (M1), vs GOE.

Honest structure:
  PART B (accuracy diagnostic): solve ONE surface at two mesh resolutions and
     compare eigenvalues mode-by-mode.  The unfolded mean spacing is ~1
     (Weyl, area=4pi), so once |lambda_n(fine)-lambda_n(coarse)| approaches ~1
     the spacing statistics at that mode are FEM-dominated.  This fixes the
     usable window [n_lo, n_hi].
  PART A: ensemble spacing histogram in that window, unfolded (xi=lambda since
     area=4pi), pooled over surfaces, vs GOE Wigner surmise and Poisson.

We deliberately use "fat" surfaces (cuffs in [1.5,4]) so the low spectrum is
generic (no tiny lambda_1 from a thin neck).  Everything is screening-grade:
the point is a first, honest look -- a decisive GOE test needs the spectral/
MPS method (many accurate modes), which is exactly where higher-order methods
become necessary.
"""
import argparse
from pathlib import Path
import numpy as np, time
from scipy.sparse import block_diag, coo_matrix
from scipy.sparse.linalg import eigsh
try:
    from ..pants import build_pants
except ImportError:  # direct script execution from validation/exploratory
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pants import build_pants
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def glue_spectrum(l1,l2,l3, tw, m, h, k):
    P=build_pants(l1,l2,l3,m=m,h=h)
    N=len(P['coords'])
    S=block_diag([P['S'],P['S']]).tocsr(); M=block_diag([P['M'],P['M']]).tocsr()
    parent=list(range(2*N))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[ra]=rb
    Kc=len(P['cuffs'][0][0])
    for i in range(3):
        chain=P['cuffs'][i][0]; j0=int(round(tw[i]*Kc))%Kc
        for kk in range(Kc): union(chain[kk], N+chain[(j0-kk)%Kc])
    roots=[find(x) for x in range(2*N)]; uniq=sorted(set(roots))
    rm={r:i for i,r in enumerate(uniq)}
    dof=np.array([rm[roots[x]] for x in range(2*N)]); nd=len(uniq)
    R=coo_matrix((np.ones(2*N),(np.arange(2*N),dof)),shape=(2*N,nd)).tocsr()
    Sd=(R.T@S@R).tocsr(); Md=(R.T@M@R).tocsr()
    md=np.asarray(Md.diagonal()).ravel(); good=np.where(md>1e-12)[0]
    Sd=Sd[good][:,good]; Md=Md[good][:,good]
    vals,_=eigsh(Sd,k=k,M=Md,sigma=-1e-6,which='LM')
    return np.sort(vals.real)

def main():
    ap = argparse.ArgumentParser(
        description="Exploratory nearest-neighbour spacing screen for the M1 genus-2 model."
    )
    ap.add_argument("--surfaces", type=int, default=45)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--k", type=int, default=45)
    ap.add_argument("--out-dir", default="validation_output")
    ap.add_argument("--error-cutoff", type=float, default=0.30)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_path = out_dir / "goe_diag.png"
    spacing_path = out_dir / "goe_spacing.png"

    # PART B: determine a resolution-limited window on one fixed surface.
    print("PART B: accuracy diagnostic (one surface, two resolutions)")
    tw0=(0.3,0.6,0.15)
    k=args.k
    lo = glue_spectrum(2.2,2.8,3.3, tw0, m=16, h=0.045, k=k)
    hi = glue_spectrum(2.2,2.8,3.3, tw0, m=28, h=0.026, k=k)
    n=np.arange(1,k)
    dlam=np.abs(lo[1:k]-hi[1:k])
    usable=np.where(dlam < args.error_cutoff)[0]
    n_hi = (usable.max()+1) if len(usable) else 6
    n_lo = 4
    if n_hi <= n_lo + 1:
        raise RuntimeError(
            "No useful spacing window survived the mesh-resolution diagnostic. "
            "Refine the meshes or relax --error-cutoff for exploratory use."
        )
    print(f"  |dlambda_n| < {args.error_cutoff:g} through approximately mode n={n_hi}")
    print(f"  => exploratory spacing window: modes [{n_lo}, {n_hi}]")

    fig,ax=plt.subplots(figsize=(6.2,4))
    ax.semilogy(n, dlam, 'o-', ms=3, color="#c33", label=r"$|\lambda_n^{fine}-\lambda_n^{coarse}|$")
    ax.axhline(1.0, color="navy", ls="--", lw=1, label="leading Weyl mean spacing ~ 1")
    ax.axhline(args.error_cutoff, color="gray", ls=":", lw=1, label=f"screening cutoff {args.error_cutoff:g}")
    ax.axvspan(n_lo, n_hi, color="#3b6", alpha=.15, label=f"screening window [{n_lo},{n_hi}]")
    ax.set_xlabel("mode index n"); ax.set_ylabel("absolute coarse/fine difference")
    ax.set_title("Resolution diagnostic for spacing statistics")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(diag_path,dpi=130)
    print(f"  saved -> {diag_path}")

    # PART A: pooled spacings.  This is intentionally only screening-grade.
    print("\nPART A: exploratory level spacings vs GOE (screening window only)")
    rng=np.random.default_rng(args.seed)
    spac=[]; t0=time.time()
    for surf in range(args.surfaces):
        L=rng.uniform(1.5,4.0,3); tw=rng.uniform(0,1,3)
        v=glue_spectrum(L[0],L[1],L[2], tw, m=24, h=0.030, k=k)
        # Leading Weyl unfolding only: for genus 2, Area=4*pi, so N(lambda)~lambda.
        # This is not a precision unfolding of the low spectrum and is why this script
        # remains exploratory rather than a paper-grade GOE test.
        xi=v[1:][:n_hi]
        d=np.diff(xi)[n_lo-1:]
        spac.extend(d)
        if (surf+1)%10==0:
            print(f"  {surf+1}/{args.surfaces} ({time.time()-t0:.0f}s)")
    spac=np.asarray(spac, dtype=float)
    if len(spac) == 0 or not np.isfinite(spac).all() or spac.mean() <= 0:
        raise RuntimeError("Invalid or empty spacing sample.")
    raw_mean=spac.mean()
    spac=spac/raw_mean
    print(f"  pooled spacings: {len(spac)}; raw mean spacing={raw_mean:.3f}")

    frac_small=np.mean(spac<0.5)
    print(f"  fraction spacings < 0.5: {frac_small:.2f}  "
          f"(screening diagnostic only; do not interpret as a universality test)")

    ss=np.linspace(0,3.5,200)
    goe=(np.pi/2)*ss*np.exp(-np.pi*ss**2/4)
    poi=np.exp(-ss)
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    ax.hist(spac,bins=np.linspace(0,3.5,22),density=True,color="#8ac",
            edgecolor="k",alpha=.8,label=f"M1 genus-2 ({len(spac)} spacings)")
    ax.plot(ss,goe,'r-',lw=2,label="GOE Wigner surmise")
    ax.plot(ss,poi,'k--',lw=1.5,label="Poisson")
    ax.set_xlabel("normalised spacing s"); ax.set_ylabel("P(s)")
    ax.set_title("Nearest-neighbour spacing (exploratory / screening-grade)")
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(spacing_path,dpi=130)
    print(f"  saved -> {spacing_path}")

if __name__ == "__main__":
    main()
