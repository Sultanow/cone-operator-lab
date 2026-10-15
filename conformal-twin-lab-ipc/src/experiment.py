"""Mizhaev's polyhedron -> C4-equivariant triangulation K, circle packing h_cp, graph spectrum.
Writes src/results.json, the input of every other script. Requires numpy, scipy, shapely."""
import numpy as np
from collections import Counter,defaultdict
from scipy.optimize import minimize, least_squares
from scipy.linalg import eigh
from scipy.sparse import lil_matrix,csr_matrix
from shapely.geometry import Polygon, Point
from shapely.ops import triangulate
import json
from pathlib import Path

V=np.array([
[-72,84,18],[-36,112,102],[72,-84,18],[36,-112,102],[0,300,234],[-84,48,-18],[9,147,207],[0,-300,234],[84,-48,-18],[-112,-36,-102],[48,84,18],[-147,9,-207],[-9,-147,207],[84,72,-18],[112,36,-102],[-48,-84,18],[147,-9,-207],[-18,126,144],[-126,-18,-144],[-300,0,-234],[-84,-72,-18],[18,-126,144],[126,18,-144],[300,0,-234]],dtype=float)
F=[
[6,10,16,22,18,7,5,1,2],
[9,15,11,18,22,13,8,3,4],
[7,5,8,3,17,23,9,15,14],
[13,8,5,1,12,19,6,10,21],
[1,2,11,18,7,14,24,20,12],
[3,4,16,22,13,21,20,24,17],
[14,24,17,23,19,6,2,11,15],
[10,21,20,12,19,23,9,4,16]]
F=[[j-1 for j in face] for face in F]
T=np.array([[0,1,0],[-1,0,0],[0,0,-1]],dtype=float)

def triangle_area2(a,b,c): return float(np.linalg.det(np.stack([b-a,c-a])))

def earclip(f):
  points=V[f]; pp=points-points[0]
  # select spanning nonparallel axes via SVD
  _,s,vh=np.linalg.svd(pp,full_matrices=False)
  co=pp@vh[:2].T
  pg=Polygon(co)
  assert pg.is_valid and pg.area>0, (f,pg.is_valid,pg.area)
  orientation=np.sign(pg.exterior.is_ccw*2-1)
  if orientation==-1: # note pg coords retain order
   pass
  face_area=np.sum(co[:,0]*np.roll(co[:,1],-1)-co[:,1]*np.roll(co[:,0],-1))/2
  sign=np.sign(face_area)
  def solve(available):
   if len(available)==3:
    a,b,c=available
    if sign*float(triangle_area2(co[a],co[b],co[c]))>1e-6:
     return [(a,b,c)]
    return None
   candidate=[]
   for ii in range(len(available)):
    a,b,c=(available[(ii-1)%len(available)],available[ii],available[(ii+1)%len(available)])
    cross=float(triangle_area2(co[a],co[b],co[c]))
    if sign*cross < 1e-6: continue
    ear=Polygon(co[[a,b,c]])
    if not pg.buffer(1e-7).covers(ear): continue
    if any(ear.buffer(1e-6).covers(Point(co[q])) for q in available if q not in (a,b,c)):continue
    remaining=available[:ii]+available[ii+1:]
    rem_pg=Polygon(co[remaining])
    if not rem_pg.is_valid or rem_pg.area<1e-7:continue
    old_pg=Polygon(co[available])
    if abs(old_pg.area-ear.area-rem_pg.area)>1e-5: continue
    L=np.array([np.linalg.norm(co[a]-co[b]),np.linalg.norm(co[b]-co[c]),np.linalg.norm(co[c]-co[a])]);qual=ear.area/max(L)**2
    candidate.append((qual,ii,(a,b,c)))
   for _,ii,t in sorted(candidate,reverse=True):
    sol=solve(available[:ii]+available[ii+1:])
    if sol is not None:return [t]+sol
   return None
  ids=solve(list(range(len(f))))
  if ids is None:raise ValueError(('No triangulation',f))
  triangles=[tuple(f[k] for k in t) for t in ids]
  assert len(triangles)==len(f)-2
  tot=sum(np.linalg.norm(np.cross(V[b]-V[a],V[c]-V[a]))/2 for a,b,c in triangles)
  assert abs(tot-pg.area)<1e-5,(pg.area,tot)
  return triangles

perm=np.array([np.argmin(np.linalg.norm(V-(T@v),axis=1)) for v in V])
# Choose 2 face representatives, and transport triangles across each C4 face orbit.
triangles=[]
for fid in [0,2]:
  seed=earclip(F[fid]); current=seed
  for j in range(4):
    triangles+=current
    current=[tuple(perm[x] for x in t) for t in current]
tri=np.array(triangles,dtype=int)

edges=sorted({tuple(sorted((int(a),int(b)))) for t in tri for a,b in [(t[0],t[1]),(t[1],t[2]),(t[2],t[0])]})
E=np.array(edges,dtype=int)
print('V E F chi genus:',len(V),len(E),len(tri),len(V)-len(E)+len(tri),1-(len(V)-len(E)+len(tri))/2)
print('edge incidences',Counter(Counter(tuple(sorted((int(a),int(b)))) for t in tri for a,b in [(t[0],t[1]),(t[1],t[2]),(t[2],t[0])]).values()))
print('T det',np.linalg.det(T), 'max vertex match',max(min(np.linalg.norm(V@(T.T)-v,axis=1)) for v in V))

# polygon angles curvature
angles=np.zeros(len(V))
for f in F:
 pts=V[f]; _,s,vh=np.linalg.svd(pts-pts[0],full_matrices=False); c=(pts-pts[0])@vh[:2].T
 signed=np.sign(sum(c[:,0]*np.roll(c[:,1],-1)-c[:,1]*np.roll(c[:,0],-1)))
 for i,k in enumerate(f):
  a=c[(i-1)%len(f)]-c[i];b=c[(i+1)%len(f)]-c[i]
  theta=np.arccos(np.clip(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b),-1,1))
  cross=triangle_area2(c[(i-1)%len(f)],c[i],c[(i+1)%len(f)])
  if cross*signed<0: theta=2*np.pi-theta
  angles[k]+=theta
K=2*np.pi-angles
print('K pos neg sum',sum(K>0),sum(K<0),K.sum(),-8*np.pi)
print('K',np.round(K,5))

# construct exact hyperbolic metric by vertex circle radii, d_ij=r_i+r_j, angles via hyperbolic law of cosines.
# By standard circle packing, a zero-curvature solution exists for triangulation with genus>1.

def hyparg(a,b,c):
  # cos angle opposing a; b,c adjacent
  return np.clip((np.cosh(b)*np.cosh(c)-np.cosh(a))/(np.sinh(b)*np.sinh(c)),-1,1)

def hyperbolic_triangle_angles(side):
 a,b,c=side
 return np.arccos(hyparg(a,b,c)),np.arccos(hyparg(b,c,a)),np.arccos(hyparg(c,a,b))

idx_tri=tri

def hyp_geometry(logr, need_area=False):
 r=np.exp(logr)
 # triangle sides across v0,v1,v2; a=r1+r2 etc
 a=r[idx_tri[:,1]]+r[idx_tri[:,2]]
 b=r[idx_tri[:,0]]+r[idx_tri[:,2]]
 c=r[idx_tri[:,0]]+r[idx_tri[:,1]]
 # arccos can lose precision small lengths; ensure bounded
 A=np.arccos(hyparg(a,b,c))
 B=np.arccos(hyparg(b,c,a))
 C=np.arccos(hyparg(c,a,b))
 vv=np.zeros(len(V))
 np.add.at(vv,tri[:,0],A)
 np.add.at(vv,tri[:,1],B)
 np.add.at(vv,tri[:,2],C)
 if need_area:
  return vv, (np.pi-A-B-C).sum(),np.stack([A,B,C],axis=1), (a,b,c)
 return vv-2*np.pi

for rs in [0.05,.1,.2,.4,.8,1.2]:
 print('circle guess',rs,'curvature range', np.round(hyp_geometry(np.log(np.full(24,rs)),True)[0].min(),3),np.round(hyp_geometry(np.log(np.full(24,rs)),True)[0].max(),3))

start=np.log(np.full(len(V),.4))
res=least_squares(hyp_geometry,start,verbose=0,method='trf',max_nfev=1500,xtol=1e-13,ftol=1e-13,gtol=1e-13)
rr=np.exp(res.x); angle_sums,area,tri_angles,sides=hyp_geometry(res.x,True)
print('packing converged',res.success,res.cost, 'residual', max(abs(angle_sums-2*np.pi)), 'radii range',rr.min(),rr.max(),'area',area)

# Symmetry: triangulation may break symmetry. Compare radii after T action; if T asymmetric we might redo symmetric triangulation.
perm=np.array([np.argmin(np.linalg.norm(V-(T@v),axis=1)) for v in V])
print('symmetry radii diff max',max(abs(rr-rr[perm])))
print('orbit map',perm+1)

# generalized vertex FEM on constant curvature hyperbolic triangles is nontrivial.
# Use consistent graph Laplacian with edge weights computed via hyperbolic circle packing combinatorial geometry.
# use symmetric normalized weights w=1/sinh(d_ij) positive and mass from hyperbolic triangle areas /3.
d_len=rr[E[:,0]]+rr[E[:,1]]
W=1/np.sinh(d_len)
L=np.zeros((len(V),len(V)))
for (i,j),w in zip(E,W):
 L[i,i]+=w;L[j,j]+=w;L[i,j]-=w;L[j,i]-=w
# hyperbolic triangle areas
tri_area=np.pi-tri_angles.sum(axis=1)
mass=np.zeros(len(V))
for pos in range(3):np.add.at(mass,tri[:,pos],tri_area/3)
print('mass sum',mass.sum(),area)
val,U=eigh(L,np.diag(mass))
print('first eigs',val[:12])
# mass orthonormal and interpolation eigenvectors; normalized V mean, exact recovery at N24.
X=V.copy()
coefs=U.T@(mass[:,None]*X)
coords={}
for n in [1,3,4,6,8,10,12,16,20,24]:
 recon=U[:,:n]@coefs[:n]
 rms=np.sqrt(np.mean(np.sum((recon-X)**2,axis=1)))
 rel=rms/np.sqrt(np.mean(np.sum((X-X.mean(axis=0))**2,axis=1)))
 coords[n]=dict(rms=float(rms),rel=float(rel),max=float(np.max(np.linalg.norm(recon-X,axis=1))))
 print('n',n,'rmse',rms,'rel',rel)

# track symmetry accuracy on recon, spectra from non invariant triang
for n in [4,8,12,16,24]:
 xr=U[:,:n]@coefs[:n]
 symmetry=np.sqrt(np.mean(np.sum((xr[perm]-xr@T.T)**2,axis=1)))
 print('symm',n,symmetry)

heat_results={}
for t in [0.005,0.02,0.05,0.10,0.20]:
  smoothed=U@(np.exp(-t*val)[:,None]*coefs)
  inv=U@(np.exp(t*val)[:,None]*(U.T@(mass[:,None]*smoothed)))
  smerr=np.sqrt(np.mean(np.sum((smoothed-X)**2,axis=1)))
  inverr=np.max(np.linalg.norm(inv-X,axis=1))
  symerr=np.max(np.linalg.norm(smoothed[perm]-smoothed@T.T,axis=1))
  heat_results[str(t)]=dict(smooth_rmse=float(smerr),smooth_relative=float(smerr/np.sqrt(np.mean(np.sum((X-X.mean(axis=0))**2,axis=1)))),inverse_max_error=float(inverr),symmetry_max_error=float(symerr),condition_number=float(np.exp(t*val[-1])))
  print('HEAT',t,heat_results[str(t)])
result=dict(heat=heat_results,vertices=V.tolist(),faces=[[x+1 for x in f] for f in F],triangles=(tri+1).tolist(),edges=(E+1).tolist(),original_curvature=K.tolist(),radii=rr.tolist(),hyp_angles=angle_sums.tolist(),hyperbolic_area=float(area),hyperbolic_residual=float(max(abs(angle_sums-2*np.pi))),mass=mass.tolist(),eigenvalues=val.tolist(),coefficients=coefs.tolist(),recon_errors=coords,triangles_area=tri_area.tolist(),symmetry_radii_max=float(max(abs(rr-rr[perm]))),symmetry_perm=(perm+1).tolist())
with open(Path(__file__).resolve().parent / 'results.json','w') as f:json.dump(result,f,indent=2)
