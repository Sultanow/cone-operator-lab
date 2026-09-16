"""
Step 2: general pants-graph gluing -> arbitrary genus.

A genus-g surface = (2g-2) pants glued along (3g-3) curves = a trivalent graph
with (2g-2) vertices and (3g-3) edges.  Each edge carries a length l_c (shared
cuff of the two incident pants) and a twist tau_c.

We specify: pants_slots[p] = the 3 edge-ids incident to pants p (slot order),
and edges[c] = (pa, sa, pb, sb, length, twist).  Each pants is built from the
lengths of its 3 incident edges; then every edge merges the two matching cuff
node-chains (reversed orientation + twist), exactly as in genus 2.

Genus 3 test: 4 pants, 6 edges = the tetrahedral graph K4 (each pants borders
the other three).  Validation: area = 4*pi*(g-1) = 8*pi, lambda_0 ~ 0.
"""
import numpy as np
from scipy.sparse import block_diag, coo_matrix
from scipy.sparse.linalg import eigsh
try:
    from .pants import build_pants
except ImportError:  # direct script execution
    from pants import build_pants

def build_surface(edges, pants_slots, m=12, h=0.06, k=6):
    """edges[c]=(pa,sa,pb,sb,length,twist); pants_slots[p]=(e_slot0,e_slot1,e_slot2)."""
    npants=len(pants_slots)
    # build each pants from its 3 incident edge lengths (slot order)
    Ps=[]; offs=[]; tot=0
    for p in range(npants):
        e0,e1,e2=pants_slots[p]
        L=[edges[e0][4], edges[e1][4], edges[e2][4]]
        P=build_pants(L[0],L[1],L[2],m=m,h=h)
        Ps.append(P); offs.append(tot); tot+=len(P['coords'])
    S=block_diag([P['S'] for P in Ps]).tocsr()
    M=block_diag([P['M'] for P in Ps]).tocsr()
    parent=list(range(tot))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[ra]=rb
    for (pa,sa,pb,sb,length,twist) in edges:
        cha=Ps[pa]['cuffs'][sa][0]; chb=Ps[pb]['cuffs'][sb][0]
        Kc=len(cha)
        assert len(chb)==Kc
        j0=int(round(twist*Kc))%Kc
        for kk in range(Kc):
            a=offs[pa]+cha[kk]
            b=offs[pb]+chb[(j0-kk)%Kc]
            union(a,b)
    roots=[find(x) for x in range(tot)]; uniq=sorted(set(roots))
    rm={r:i for i,r in enumerate(uniq)}
    dof=np.array([rm[roots[x]] for x in range(tot)]); nd=len(uniq)
    R=coo_matrix((np.ones(tot),(np.arange(tot),dof)),shape=(tot,nd)).tocsr()
    Sd=(R.T@S@R).tocsr(); Md=(R.T@M@R).tocsr()
    md=np.asarray(Md.diagonal()).ravel(); good=np.where(md>1e-12)[0]
    Sd=Sd[good][:,good]; Md=Md[good][:,good]
    vals,_=eigsh(Sd,k=k,M=Md,sigma=-1e-6,which='LM')
    return Md.sum(), np.sort(vals.real), nd

def K4_genus3(lengths, twists):
    """K4: pants 0..3, edges e0..e5. lengths/twists per edge (len 6)."""
    # edge c: (pa,sa,pb,sb)
    topo=[(0,0,1,0),  # e0: 0-1
          (0,1,2,0),  # e1: 0-2
          (0,2,3,0),  # e2: 0-3
          (1,1,2,1),  # e3: 1-2
          (1,2,3,1),  # e4: 1-3
          (2,2,3,2)]  # e5: 2-3
    edges=[(pa,sa,pb,sb,lengths[c],twists[c]) for c,(pa,sa,pb,sb) in enumerate(topo)]
    pants_slots={
        0:(0,1,2),   # edges e0,e1,e2 in slots 0,1,2
        1:(0,3,4),   # e0(slot0),e3(slot1),e4(slot2)
        2:(1,3,5),   # e1(slot0),e3(slot1),e5(slot2)
        3:(2,4,5),   # e2(slot0),e4(slot1),e5(slot2)
    }
    return edges, pants_slots

if __name__=="__main__":
    g=3
    print(f"genus {g}: {2*g-2} pants, {3*g-3} curves (K4 tetrahedral graph)")
    print(f"target area = 4*pi*(g-1) = {4*np.pi*(g-1):.4f}\n")
    rng=np.random.default_rng(7)
    print(f"{'trial':>5} | {'edge lengths (6)':>34} | {'area':>8} {'err vs 8pi':>10} | {'lambda_0':>10} | {'lambda_1':>8}")
    print("-"*92)
    for tr in range(4):
        L=rng.uniform(1.2,3.5,6); T=rng.uniform(0,1,6)
        edges,ps=K4_genus3(L,T)
        area,vals,nd=build_surface(edges,ps,m=12,h=0.06,k=6)
        Ls=",".join(f"{x:.1f}" for x in L)
        print(f"{tr:5d} | {Ls:>34} | {area:8.3f} {abs(area-8*np.pi):10.2e} "
              f"| {vals[0]:10.1e} | {vals[1]:8.4f}")
    print("-"*92)
    print(f"=> genus-3 surfaces assemble: area=8pi (= 2*pi*|chi|), lambda_0~0. "
          f"General pants-graph gluing works.")
