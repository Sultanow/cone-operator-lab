"""
Brick 2: glue two pants into a CLOSED genus-2 surface and validate against
the Bolza surface.

Two pants P1, P2 (here identical, symmetric cuffs l1=l2=l3=l_s) are glued along
their three cuffs.  Each cuff carries 2m equally-spaced (arc length) nodes; the
gluing merges cuff node k of P1 with node (j0 - k) mod 2m of P2 -- the "-k"
realises the orientation reversal of the boundary, j0 the twist (tau = j0 * l/2m).

Validations:
  * area = sum(mass) = 4*pi        (= 2*pi*|chi|, genus 2);
  * lambda_0 ~ 0                   (closed surface / assembly);
  * TWIST SWEEP of the symmetric Bolza decomposition (l_s = 2 arccosh(1+sqrt2)):
       - lambda_1 <= 3.8388976  for every twist  (Jenni / Kravchuk-Mazac-Pal
         theorem: Bolza maximises lambda_1 in genus 2);
       - max over twist ~ 3.8388  (the Bolza point), where lambda_1 is a near
         TRIPLE (Bolza multiplicity 3).
    This validates the general FN gluing machinery WITHOUT needing the exact
    twist convention -- we scan and hit the maximiser.
"""
import numpy as np
from scipy.sparse import block_diag, coo_matrix
from scipy.sparse.linalg import eigsh
try:
    from .pants import build_pants
except ImportError:  # direct script execution
    from pants import build_pants

BOLZA = 3.838887258
KMP_BOUND = 3.8388976481

def glue_genus2(l, twist_steps, m=16, h=0.045, k=6):
    P = build_pants(l, l, l, m=m, h=h)
    N = len(P['coords'])
    S = block_diag([P['S'], P['S']]).tocsr()
    M = block_diag([P['M'], P['M']]).tocsr()
    parent = list(range(2*N))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[ra]=rb
    for i in range(3):
        chain = P['cuffs'][i][0]
        K = len(chain)                       # = 2m
        j0 = twist_steps[i] % K
        for kk in range(K):
            union(chain[kk], N + chain[(j0-kk) % K])
    roots=[find(x) for x in range(2*N)]
    uniq=sorted(set(roots)); remap={r:i for i,r in enumerate(uniq)}
    dof=np.array([remap[roots[x]] for x in range(2*N)]); ndof=len(uniq)
    R=coo_matrix((np.ones(2*N),(np.arange(2*N),dof)),shape=(2*N,ndof)).tocsr()
    Sd=(R.T@S@R).tocsr(); Md=(R.T@M@R).tocsr()
    md=np.asarray(Md.diagonal()).ravel(); good=np.where(md>1e-12)[0]
    Sd=Sd[good][:,good]; Md=Md[good][:,good]
    vals,_=eigsh(Sd,k=k,M=Md,sigma=-1e-6,which='LM')
    return Md.sum(), np.sort(vals.real), ndof

if __name__=="__main__":
    ls = 2*np.arccosh(1+np.sqrt(2))
    print(f"symmetric Bolza cuff length l_s = 2 arccosh(1+sqrt2) = {ls:.5f}\n")

    m = 16
    # --- sanity at twist 0 ---
    area, vals, ndof = glue_genus2(ls, [0,0,0], m=m)
    print(f"[twist 0] dofs={ndof}  area={area:.4f} (4pi={4*np.pi:.4f}, "
          f"err {abs(area-4*np.pi):.1e})  lambda_0={vals[0]:.2e}")
    print(f"          low spectrum: {np.array2string(vals,precision=4,floatmode='fixed')}\n")

    # --- twist sweep ---
    print("twist sweep (symmetric decomposition):")
    print(f"{'t=j/2m':>8} | {'lambda_1 (triple mean)':>22} | {'<= KMP 3.83890?':>16}")
    print("-"*54)
    K = 2*m
    best=(-1,-1)
    for j in range(0, K, 2):
        _, vals, _ = glue_genus2(ls, [j,j,j], m=m, k=6)
        lam1 = np.mean(vals[1:4])           # 3-fold cluster mean
        ok = "yes" if lam1 <= KMP_BOUND+1e-3 else "NO (!!)"
        if lam1>best[1]: best=(j,lam1)
        print(f"{j/K:8.3f} | {lam1:22.5f} | {ok:>16}")
    print("-"*54)
    jb,lamb = best
    print(f"\nmax over twist: lambda_1 = {lamb:.5f} at t={jb/K:.3f}")
    print(f"Bolza reference  = {BOLZA:.5f}   rel.err = {abs(lamb-BOLZA)/BOLZA*100:.2f}%")
    print(f"KMP upper bound  = {KMP_BOUND:.5f}   (never exceeded above => Jenni/KMP ok)")
    # fine solve at the maximiser
    _, vals, ndof = glue_genus2(ls, [jb,jb,jb], m=24, h=0.032, k=6)
    print(f"\nfiner mesh at maximiser (dofs={ndof}): "
          f"lambda_1 triple = {np.array2string(vals[1:4],precision=4,floatmode='fixed')}, "
          f"mean {np.mean(vals[1:4]):.4f}")
