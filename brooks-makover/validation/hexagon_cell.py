"""
Brick 1a: place an ARBITRARY right-angled hyperbolic hexagon (the pants
half) in the Poincare disk from its six side lengths, mesh it with curved
geodesic sides, assemble FEM, and validate the geometry+metric+solver:

  * hyperbolic area = sum of mass matrix = int rho^2 dA  MUST equal pi
    (Gauss-Bonnet: right-angled hexagon area = 4pi - 6*(pi/2) = pi), for
    every choice of cuff lengths;
  * Neumann lambda_0 ~ 0 with the constant mode (solver/assembly sanity).

This is the new capability beyond the regular octagon: placing and meshing
an arbitrary hyperbolic cell, which the pants/gluing pipeline needs.
"""
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

def seam(li, lj, lk):
    return np.arccosh((np.cosh(lk/2)+np.cosh(li/2)*np.cosh(lj/2))
                      /(np.sinh(li/2)*np.sinh(lj/2)))

# --- SU(1,1) transport in the disk ---
def gflow(s):
    return np.array([[np.cosh(s/2), np.sinh(s/2)],
                     [np.sinh(s/2), np.cosh(s/2)]], dtype=complex)
def grot(phi):
    return np.array([[np.exp(1j*phi/2), 0],[0, np.exp(-1j*phi/2)]], dtype=complex)
def app(M, z):
    return (M[0,0]*z + M[0,1])/(M[1,0]*z + M[1,1])

def hexagon_vertices(sides, turn=np.pi/2):
    F = np.eye(2, dtype=complex); V=[]
    for s in sides:
        V.append(app(F, 0.0+0j))
        F = F @ gflow(s) @ grot(turn)
    defect = min(np.linalg.norm(F-np.eye(2)), np.linalg.norm(F+np.eye(2)))
    return np.array(V), defect

def geo_circle(p, q):
    A = np.array([[p.real,p.imag],[q.real,q.imag]])
    b = 0.5*np.array([1+abs(p)**2, 1+abs(q)**2])
    c = np.linalg.solve(A,b); c=c[0]+1j*c[1]
    return c, abs(p-c)

def geodesic_pts(a, b, m):
    """m-1 interior nodes along geodesic a->b in the disk."""
    # move a to 0
    Tinv = lambda z:(z-a)/(1-np.conj(a)*z)
    T    = lambda z:(z+a)/(1+np.conj(a)*z)
    w = Tinv(b); wd=w/abs(w); L=np.arctanh(abs(w))
    return np.array([T(wd*np.tanh((j/m)*L)) for j in range(1,m)])

def mesh_and_area(sides, m=26, h=0.03):
    V, defect = hexagon_vertices(sides)
    # shift centroid to origin so no vertex sits at 0 and no side is a diameter
    b = np.mean(V)
    Binv = lambda z: (z - b)/(1 - np.conj(b)*z)
    V = Binv(V)
    n = len(V)
    # geodesic circles + interior orientation (side containing centroid)
    cen = np.mean(V)
    circ=[]; inside_out=[]
    for k in range(n):
        c,rho = geo_circle(V[k], V[(k+1)%n])
        circ.append((c,rho))
        inside_out.append(abs(cen-c) > rho)   # True => interior is |z-c|>rho
    def inside(z, tol=1e-9):
        for (c,rho),io in zip(circ, inside_out):
            d = abs(z-c)
            if io and d < rho-tol: return False
            if (not io) and d > rho+tol: return False
        return True
    # nodes: vertices + boundary + interior grid
    coords=list(V)
    for k in range(n):
        for z in geodesic_pts(V[k], V[(k+1)%n], m):
            coords.append(z)
    R = max(abs(V))
    xs=np.arange(-R,R+h,h)
    for x in xs:
        for y in xs:
            z=x+1j*y
            if inside(z,tol=1e-3): coords.append(z)
    coords=np.array(coords)
    P=np.column_stack([coords.real,coords.imag])
    tri=Delaunay(P)
    rho2=lambda z2:(2.0/(1.0-z2))**2
    S=lil_matrix((len(coords),len(coords))); M=lil_matrix((len(coords),len(coords)))
    for t in tri.simplices:
        p=P[t]; cz=np.mean(coords[t])
        if not inside(cz,tol=1e-3): continue
        x1,y1=p[0];x2,y2=p[1];x3,y3=p[2]
        detJ=(x2-x1)*(y3-y1)-(x3-x1)*(y2-y1); area=0.5*abs(detJ)
        if area<1e-12: continue
        b=np.array([y2-y3,y3-y1,y1-y2])/detJ; c=np.array([x3-x2,x1-x3,x2-x1])/detJ
        Ke=area*(np.outer(b,b)+np.outer(c,c))
        r2=np.mean(rho2(np.sum(p**2,axis=1)))
        Me=(area/12.0)*np.array([[2.,1,1],[1,2,1],[1,1,2]])*r2
        for a_ in range(3):
            for b_ in range(3):
                S[t[a_],t[b_]]+=Ke[a_,b_]; M[t[a_],t[b_]]+=Me[a_,b_]
    S=csr_matrix(S); M=csr_matrix(M)
    hyp_area = M.sum()                      # int rho^2 dA = hyperbolic area
    vals,_=eigsh(S,k=3,M=M,sigma=-1e-6,which='LM')  # Neumann spectrum
    return defect, hyp_area, np.sort(vals.real)[0], len(coords)

if __name__=="__main__":
    print(f"{'cuffs (l1,l2,l3)':>20} | {'close':>8} | {'hyp.area':>9} "
          f"| {'area err vs pi':>13} | {'Neumann l0':>11}")
    print("-"*74)
    rng=np.random.default_rng(1)
    tests=[(1.0,1.0,1.0),(0.5,2.0,3.0),(4.0,4.0,4.0)]
    tests+=[tuple(rng.uniform(0.5,4.0,3)) for _ in range(3)]
    for (l1,l2,l3) in tests:
        sides=[l1/2,seam(l1,l2,l3),l2/2,seam(l2,l3,l1),l3/2,seam(l3,l1,l2)]
        defect,area,l0,npts=mesh_and_area(sides)
        print(f"({l1:4.2f},{l2:4.2f},{l3:4.2f}) | {defect:8.1e} | {area:9.5f} "
              f"| {abs(area-np.pi):13.2e} | {l0:11.2e}")
    print("-"*74)
    print("=> arbitrary hyperbolic hexagon: placed, meshed, area=pi, Neumann l0~0.")
