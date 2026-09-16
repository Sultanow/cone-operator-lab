"""
Brick 1b: the hyperbolic PAIR OF PANTS as the double of the right-angled
hexagon across its three seams.

Two isometric hexagon copies A, B share the three SEAM boundaries (identified
node-by-node); the three CUFF boundaries stay free, each a closed geodesic of
length l_i formed by A's half-cuff + B's half-cuff.  FEM matrices are intrinsic,
so B may reuse A's disk coordinates (an isometric overlay) -- only the abstract
gluing (shared DOFs) matters.

Validations (all independent of eigen-answers):
  * hyperbolic area = sum(mass) = 2*pi     (two hexagons of area pi);
  * Neumann lambda_0 ~ 0                    (assembly/solver sanity);
  * each reconstructed cuff length ~ l_i    (arc-length parametrization ok).
The arc-length cuff node lists are returned, ready for pants-to-pants gluing
(merge cuff DOFs with a twist offset) in the next brick.
"""
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

def seam(li,lj,lk):
    return np.arccosh((np.cosh(lk/2)+np.cosh(li/2)*np.cosh(lj/2))
                      /(np.sinh(li/2)*np.sinh(lj/2)))
def gflow(s): return np.array([[np.cosh(s/2),np.sinh(s/2)],
                               [np.sinh(s/2),np.cosh(s/2)]],dtype=complex)
def grot(p):  return np.array([[np.exp(1j*p/2),0],[0,np.exp(-1j*p/2)]],dtype=complex)
def app(M,z): return (M[0,0]*z+M[0,1])/(M[1,0]*z+M[1,1])
def hdist(z,w):
    return np.arccosh(1+2*abs(z-w)**2/((1-abs(z)**2)*(1-abs(w)**2)))

def hexagon(sides, m=26, h=0.03):
    F=np.eye(2,dtype=complex); V=[]
    for s in sides:
        V.append(app(F,0j)); F=F@gflow(s)@grot(np.pi/2)
    V=np.array(V); b=np.mean(V); Binv=lambda z:(z-b)/(1-np.conj(b)*z); V=Binv(V)
    n=len(V); cen=np.mean(V)
    circ=[]; io=[]
    for k in range(n):
        p,q=V[k],V[(k+1)%n]
        A=np.array([[p.real,p.imag],[q.real,q.imag]]); rhs=0.5*np.array([1+abs(p)**2,1+abs(q)**2])
        c=np.linalg.solve(A,rhs); c=c[0]+1j*c[1]
        circ.append((c,abs(p-c))); io.append(abs(cen-c)>abs(p-c))
    def inside(z,tol=1e-9):
        for (c,rho),i in zip(circ,io):
            d=abs(z-c)
            if i and d<rho-tol: return False
            if (not i) and d>rho+tol: return False
        return True
    def gpts(a,b,mm):
        Ti=lambda z:(z-a)/(1-np.conj(a)*z); T=lambda z:(z+a)/(1+np.conj(a)*z)
        w=Ti(b); wd=w/abs(w); L=np.arctanh(abs(w))
        return [T(wd*np.tanh((j/mm)*L)) for j in range(1,mm)]
    coords=list(V)
    side_idx=[[] for _ in range(n)]
    for k in range(n):
        for z in gpts(V[k],V[(k+1)%n],m):
            side_idx[k].append(len(coords)); coords.append(z)
    R=max(abs(V)); xs=np.arange(-R,R+h,h)
    for x in xs:
        for y in xs:
            z=x+1j*y
            if inside(z,tol=1e-3): coords.append(z)
    coords=np.array(coords)
    P=np.column_stack([coords.real,coords.imag]); tri=Delaunay(P)
    tris=[t for t in tri.simplices if inside(np.mean(coords[t]),tol=1e-3)]
    return coords, tris, side_idx   # corners are indices 0..n-1

def assemble(coords, tris):
    rho2=lambda z2:(2.0/(1.0-z2))**2
    N=len(coords); S=lil_matrix((N,N)); M=lil_matrix((N,N))
    for t in tris:
        p=np.column_stack([coords[list(t)].real, coords[list(t)].imag])
        x1,y1=p[0];x2,y2=p[1];x3,y3=p[2]
        detJ=(x2-x1)*(y3-y1)-(x3-x1)*(y2-y1); area=0.5*abs(detJ)
        if area<1e-12: continue
        b=np.array([y2-y3,y3-y1,y1-y2])/detJ; c=np.array([x3-x2,x1-x3,x2-x1])/detJ
        Ke=area*(np.outer(b,b)+np.outer(c,c))
        r2=np.mean(rho2(np.sum(p**2,axis=1)))
        Me=(area/12.0)*np.array([[2.,1,1],[1,2,1],[1,1,2]])*r2
        tt=list(t)
        for a_ in range(3):
            for b_ in range(3):
                S[tt[a_],tt[b_]]+=Ke[a_,b_]; M[tt[a_],tt[b_]]+=Me[a_,b_]
    return csr_matrix(S), csr_matrix(M)

def build_pants(l1,l2,l3, m=26, h=0.03):
    sides=[l1/2,seam(l1,l2,l3),l2/2,seam(l2,l3,l1),l3/2,seam(l3,l1,l2)]
    coords,tris,side_idx=hexagon(sides,m,h)
    NA=len(coords)
    ncorner=6
    seam_sides=[1,3,5]; cuff_sides=[0,2,4]
    # shared A-nodes: corners + seam-side nodes
    shared=set(range(ncorner))
    for k in seam_sides: shared|=set(side_idx[k])
    # B copies for non-shared nodes
    bmap={}; extra=[]
    for i in range(NA):
        if i in shared: bmap[i]=i
        else: bmap[i]=NA+len(extra); extra.append(coords[i])
    coords_all=np.concatenate([coords, np.array(extra)]) if extra else coords.copy()
    trisB=[tuple(bmap[i] for i in t) for t in tris]
    S,M=assemble(coords_all, list(tris)+trisB)
    area=M.sum()
    vals,_=eigsh(S,k=3,M=M,sigma=-1e-6,which='LM'); l0=np.sort(vals.real)[0]
    # reconstruct full cuffs: cuff i = A half-cuff (side 2i) + B half-cuff
    #   ordered node chain along the closed cuff geodesic
    cuffs=[]
    corner_next={0:1,2:3,4:5}  # side k goes corner k -> corner k+1
    for ci,k in enumerate(cuff_sides):
        a0,a1=k,(k+1)%6                    # corner endpoints of half-cuff side k
        A_half=[a0]+side_idx[k]+[a1]       # A indices along half-cuff
        B_half=[bmap[i] for i in A_half]   # B copy (endpoints shared=corners)
        # full cuff chain: A_half then B_half reversed (share the two corners)
        chain=A_half + B_half[-2:0:-1]     # avoid repeating the two shared corners
        # arc-length
        s=[0.0]
        for j in range(1,len(chain)):
            s.append(s[-1]+hdist(coords_all[chain[j-1]], coords_all[chain[j]]))
        # close-up length (last back to first)
        Lcuff=s[-1]+hdist(coords_all[chain[-1]], coords_all[chain[0]])
        cuffs.append((chain, np.array(s), Lcuff))
    return dict(coords=coords_all,S=S,M=M,area=area,l0=l0,cuffs=cuffs,
                target=(l1,l2,l3))

if __name__=="__main__":
    print(f"{'cuffs':>18} | {'area':>8} {'err vs 2pi':>10} | {'Neu l0':>9} "
          f"| cuff-length reconstruction")
    print("-"*84)
    rng=np.random.default_rng(2)
    tests=[(1.0,1.0,1.0),(0.5,2.0,3.0),(3.0,1.2,2.5)]
    tests+=[tuple(rng.uniform(0.6,3.5,3)) for _ in range(2)]
    for (l1,l2,l3) in tests:
        r=build_pants(l1,l2,l3)
        recon=[f"{r['cuffs'][i][2]:.3f}(exp {t:.3f})" for i,t in enumerate((l1,l2,l3))]
        print(f"({l1:4.2f},{l2:4.2f},{l3:4.2f}) | {r['area']:8.4f} "
              f"{abs(r['area']-2*np.pi):10.2e} | {r['l0']:9.1e} | "
              + ", ".join(recon))
    print("-"*84)
    print("=> pair of pants: area=2pi, Neumann l0~0, cuff lengths recovered, "
          "cuffs arc-length-parametrized for gluing.")
