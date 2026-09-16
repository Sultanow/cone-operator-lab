"""
Closed genus-2 hyperbolic surface from the regular hyperbolic octagon
(opposite-side identification -> Bolza-type surface), FEM spectrum.

Geometry (Poincare disk):
  - regular octagon, interior angle pi/4 at every vertex (angle sum 8*pi/4 = 2*pi,
    so the 8 vertices glue to ONE smooth cone point of total angle 2*pi).
  - vertices V_k = 2^{-1/4} * exp(i*k*pi/4),  k=0..7.
  - side s_k joins V_k -> V_{k+1}; opposite sides paired: s_k <-> s_{k+4}.

Side-pairing isometries g_k (k=0..3), built exactly as disk automorphisms
(Blaschke composition) with orientation  g_k(V_k)=V_{k+5}, g_k(V_{k+1})=V_{k+4}.
This orientation glues all 8 vertices into a SINGLE class (checked below).

FEM uses 2D conformal invariance: stiffness = Euclidean, mass carries rho^2.
No Dirichlet BC (closed surface); identified boundary DOFs are merged.

Self-consistency tests (independent of knowing the answer):
  (T1) each g_k is a genuine disk isometry (|p|=|q| in Blaschke frames);
  (T2) g_k maps side s_k exactly onto side s_{k+4} (endpoint images match);
  (T3) the 8 vertices form a single identified class;
  (T4) lambda_0 ~ 0 with a constant eigenvector  (=> closed topology / assembly ok);
Then compare lambda_1 to the published Bolza value 3.838887.
"""
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

rv = 2.0**(-0.25)
V = np.array([rv*np.exp(1j*k*np.pi/4) for k in range(8)])

# ---- disk automorphisms -------------------------------------------------
def T(a):        # sends 0 -> a
    return lambda z: (z + a)/(1 + np.conj(a)*z)
def Tinv(a):     # sends a -> 0
    return lambda z: (z - a)/(1 - np.conj(a)*z)

def isometry_from_two(z1, w1, z2, w2):
    """orientation-preserving disk isometry with z1->w1, z2->w2 (if compatible)."""
    p = Tinv(z1)(z2)
    q = Tinv(w1)(w2)
    phi = np.angle(q) - np.angle(p)
    rot = np.exp(1j*phi)
    return (lambda z: T(w1)(rot*Tinv(z1)(z))), abs(p), abs(q)

# generators g_0..g_3 : s_k -> s_{k+4}, V_k->V_{k+5}, V_{k+1}->V_{k+4}
G = []
print("== (T1)/(T2) side-pairing isometry checks ==")
for k in range(4):
    g, ap, aq = isometry_from_two(V[k], V[(k+5)%8], V[(k+1)%8], V[(k+4)%8])
    G.append(g)
    # endpoint image check
    e1 = abs(g(V[k]) - V[(k+5)%8]); e2 = abs(g(V[(k+1)%8]) - V[(k+4)%8])
    print(f"  g_{k}: |p|-|q|={ap-aq:+.2e} (isometry), "
          f"endpoint err={max(e1,e2):.2e}")

def apply_pairing(kmap, z):     # g_k for k<4, g_{k-4}^{-1} for k>=4
    if kmap < 4:  return G[kmap](z)
    else:
        # inverse of G[kmap-4]: solve numerically via fixed formula (it's an isometry)
        raise RuntimeError

# ---- (T3) vertex identification classes ---------------------------------
parent = list(range(8))
def find(x):
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(a,b): parent[find(a)]=find(b)
for k in range(4):
    union(k, (k+5)%8)       # g_k(V_k)=V_{k+5}
    union((k+1)%8, (k+4)%8) # g_k(V_{k+1})=V_{k+4}
classes = len(set(find(i) for i in range(8)))
print(f"== (T3) vertex classes: {classes}  (need 1 for smooth genus-2) ==")

# ---- geodesic side circles (for inside-test) ----------------------------
def geo_circle(p, q):
    A = np.array([[p.real, p.imag],[q.real, q.imag]])
    b = 0.5*np.array([1+abs(p)**2, 1+abs(q)**2])
    c = np.linalg.solve(A, b)
    c = c[0] + 1j*c[1]
    rho = abs(p - c)
    return c, rho
CIRC = [geo_circle(V[k], V[(k+1)%8]) for k in range(8)]

def inside_octagon(z, tol=1e-9):
    for (c, rho) in CIRC:
        if abs(z - c) < rho - tol:   # inside a geodesic circle => beyond that side
            return False
    return True

# geodesic points from V_k to V_{k+1}
def side_nodes(k, m):
    a = V[k]; b = V[(k+1)%8]
    w = Tinv(a)(b); wdir = w/abs(w); L = np.arctanh(abs(w))
    pts=[]
    for j in range(1, m):
        t = j/m
        z0 = wdir*np.tanh(t*L)
        pts.append(T(a)(z0))
    return np.array(pts)

# ---- build node set with DOF identification -----------------------------
m = 34
coords = list(V)                     # 0..7 vertices
dof = [0]*8                          # all vertices -> dof 0
next_dof = 1
for k in range(4):
    sk = side_nodes(k, m)
    for z in sk:
        img = G[k](z)                # matching node on opposite side s_{k+4}
        coords.append(z);   dof.append(next_dof)
        coords.append(img); dof.append(next_dof)
        next_dof += 1

nb = len(coords)                     # boundary nodes count
# interior grid clipped to octagon
h = 0.026
xs = np.arange(-rv, rv+h, h)
for x in xs:
    for y in xs:
        z = x+1j*y
        if inside_octagon(z, tol=1e-3):
            coords.append(z); dof.append(next_dof); next_dof += 1
coords = np.array(coords)
dof = np.array(dof)
ndof = next_dof
print(f"nodes={len(coords)} (boundary {nb}), dofs={ndof}")

# ---- FEM assembly -------------------------------------------------------
P = np.column_stack([coords.real, coords.imag])
tri = Delaunay(P)
rho2 = lambda z2: (2.0/(1.0 - z2))**2
S = lil_matrix((len(coords), len(coords)))
M = lil_matrix((len(coords), len(coords)))
kept = 0
for t in tri.simplices:
    p = P[t]
    cz = np.mean(coords[t])
    if not inside_octagon(cz, tol=1e-3):   # trim slivers outside curved sides
        continue
    x1,y1=p[0]; x2,y2=p[1]; x3,y3=p[2]
    detJ=(x2-x1)*(y3-y1)-(x3-x1)*(y2-y1)
    area=0.5*abs(detJ)
    if area<1e-12: continue
    kept += 1
    b=np.array([y2-y3,y3-y1,y1-y2])/detJ
    c=np.array([x3-x2,x1-x3,x2-x1])/detJ
    Ke=area*(np.outer(b,b)+np.outer(c,c))
    z2v=np.sum(p**2,axis=1)
    r2=np.mean(rho2(z2v))
    Me=(area/12.0)*np.array([[2.,1,1],[1,2,1],[1,1,2]])*r2
    for a_ in range(3):
        for b_ in range(3):
            S[t[a_],t[b_]]+=Ke[a_,b_]; M[t[a_],t[b_]]+=Me[a_,b_]
print(f"kept triangles: {kept}")
S=csr_matrix(S); M=csr_matrix(M)

# ---- merge identified DOFs:  Sd = R^T S R, Md = R^T M R ------------------
from scipy.sparse import coo_matrix
R = coo_matrix((np.ones(len(coords)), (np.arange(len(coords)), dof)),
               shape=(len(coords), ndof)).tocsr()
Sd = (R.T @ S @ R).tocsr()
Md = (R.T @ M @ R).tocsr()

# drop any dof with ~zero mass (unused)
massdiag = np.asarray(Md.diagonal()).ravel()
good = np.where(massdiag > 1e-12)[0]
Sd = Sd[good][:,good]; Md = Md[good][:,good]
print(f"active dofs: {Sd.shape[0]}")

vals,_ = eigsh(Sd, k=6, M=Md, sigma=-1e-6, which='LM')
vals = np.sort(vals.real)
print("== (T4) smallest eigenvalues ==")
print("  ", np.array2string(vals, precision=5, floatmode='fixed'))
print(f"  lambda_0 = {vals[0]:.3e}   (should be ~0)")
lam1 = vals[1]
print(f"  lambda_1 = {lam1:.5f}   Bolza reference = 3.838887   "
      f"rel.err = {abs(lam1-3.838887)/3.838887*100:.2f}%")
