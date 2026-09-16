"""
Plan-A phenomenology, first pass: statistics of lambda_1 for random genus-2
surfaces in the M1 model (random pants gluing).

M1 model (genus 2, theta-graph decomposition): both pants share cuff lengths
(l1,l2,l3); we draw l_i from a fixed distribution and twists uniformly.  This
is a well-defined, tractable random surface model -- NOT the Weil-Petersson
measure (exact WP sampling is a separate hard problem).  Results are labelled
accordingly.

For each surface we build the two pants, glue, and solve the low spectrum.
We record lambda_1, the shortest cuff (a proxy upper bound for the systole),
and validate area = 4*pi and lambda_0 = 0 on every sample.

Expected/validating phenomenology (collar / Cheeger):
  a short cuff = thin neck => small Cheeger constant => small lambda_1.
So lambda_1 should correlate positively with the shortest cuff, and drop
toward 0 as the shortest cuff -> 0.  And Jenni/KMP: lambda_1 <= 3.839 (up to
the ~1% screening wobble).
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def glue_from_pants(P, twist_steps, k=6):
    N=len(P['coords'])
    S=block_diag([P['S'],P['S']]).tocsr(); M=block_diag([P['M'],P['M']]).tocsr()
    parent=list(range(2*N))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[ra]=rb
    for i in range(3):
        chain=P['cuffs'][i][0]; K=len(chain); j0=twist_steps[i]%K
        for kk in range(K): union(chain[kk], N+chain[(j0-kk)%K])
    roots=[find(x) for x in range(2*N)]; uniq=sorted(set(roots))
    remap={r:i for i,r in enumerate(uniq)}
    dof=np.array([remap[roots[x]] for x in range(2*N)]); nd=len(uniq)
    R=coo_matrix((np.ones(2*N),(np.arange(2*N),dof)),shape=(2*N,nd)).tocsr()
    Sd=(R.T@S@R).tocsr(); Md=(R.T@M@R).tocsr()
    md=np.asarray(Md.diagonal()).ravel(); good=np.where(md>1e-12)[0]
    Sd=Sd[good][:,good]; Md=Md[good][:,good]
    vals,_=eigsh(Sd,k=k,M=Md,sigma=-1e-6,which='LM')
    return Md.sum(), np.sort(vals.real)

def sample_M1(N=120, m=12, h=0.06, lo=0.4, hi=5.0, seed=0):
    rng=np.random.default_rng(seed)
    lam1=[]; mincuff=[]; areas=[]; l0s=[]; bad=0
    t0=time.time()
    for n in range(N):
        L=rng.uniform(lo,hi,3)
        try:
            P=build_pants(L[0],L[1],L[2],m=m,h=h)
        except Exception:
            bad+=1; continue
        # discretize twists to node resolution (K=2m per cuff)
        Kc=len(P['cuffs'][0][0])
        tw=[int(round(rng.uniform(0,1)*Kc))%Kc for _ in range(3)]
        area,vals=glue_from_pants(P,tw,k=6)
        lam1.append(vals[1]); mincuff.append(L.min())
        areas.append(area); l0s.append(abs(vals[0]))
        if (n+1)%20==0:
            print(f"  {n+1}/{N}  ({time.time()-t0:.0f}s)")
    return (np.array(lam1),np.array(mincuff),np.array(areas),np.array(l0s),bad)

if __name__=="__main__":
    ap = argparse.ArgumentParser(description="Exploratory genus-2 M1 phenomenology (not WP sampling).")
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--m", type=int, default=12)
    ap.add_argument("--h", type=float, default=0.06)
    ap.add_argument("--lo", type=float, default=0.4)
    ap.add_argument("--hi", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="validation_output")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "m1_stats.png"

    lam1,mincuff,areas,l0s,bad=sample_M1(
        N=args.samples, m=args.m, h=args.h, lo=args.lo, hi=args.hi, seed=args.seed
    )
    if len(lam1) == 0:
        raise RuntimeError("No valid surfaces were generated.")

    print(f"\nsamples: {len(lam1)} (failed builds: {bad})")
    print(f"validation:  area = {areas.mean():.4f} +/- {areas.std():.4f} "
          f"(4pi={4*np.pi:.4f});  max|lambda_0| = {l0s.max():.1e}")
    print(f"lambda_1:  min={lam1.min():.3f}  median={np.median(lam1):.3f}  "
          f"max={lam1.max():.3f}  (Bolza reference ~3.839)")
    over=np.mean(lam1>3.839)*100
    print(f"fraction with lambda_1>3.839 (screening/FEM wobble): {over:.0f}%")
    print(f"fraction with lambda_1<1/4: {np.mean(lam1<0.25)*100:.0f}%")
    r=np.corrcoef(mincuff,lam1)[0,1]
    print(f"corr(lambda_1, shortest cuff) = {r:+.2f}  "
          f"(short cuff is only a systole proxy in this experiment)")

    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    ax[0].hist(lam1,bins=24,color="#3b6",edgecolor="k",alpha=.8)
    ax[0].axvline(3.838887258,color="crimson",ls="--",lw=1.5,label="Bolza reference ~3.839")
    ax[0].axvline(0.25,color="navy",ls=":",lw=1.5,label="1/4")
    ax[0].set_xlabel(r"$\lambda_1$"); ax[0].set_ylabel("count")
    ax[0].set_title("M1 random genus-2 surfaces: $\\lambda_1$ distribution")
    ax[0].legend(fontsize=8)
    ax[1].scatter(mincuff,lam1,s=14,c="#37a",alpha=.7)
    ax[1].axhline(0.25,color="navy",ls=":",lw=1)
    ax[1].set_xlabel("shortest cuff length (systole proxy)")
    ax[1].set_ylabel(r"$\lambda_1$")
    ax[1].set_title("cuff length vs. first non-zero eigenvalue")
    fig.tight_layout(); fig.savefig(out_png,dpi=130)
    print(f"\nsaved plot -> {out_png}")
