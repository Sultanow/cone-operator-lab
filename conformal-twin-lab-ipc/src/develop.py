"""Develop the circle-packing hyperbolic surface into the Poincare disk and
compute the data used in the paper (selection rule, flowers, orbifold checks)."""
import json, numpy as np
from collections import deque
from scipy.linalg import eigh
import os
HERE = os.path.dirname(os.path.abspath(__file__))
R=json.load(open(os.path.join(HERE, 'results.json')))
V=np.array(R['vertices'],float); tri=np.array(R['triangles'])-1; E=np.array(R['edges'])-1
F=[[x-1 for x in f] for f in R['faces']]; rr=np.array(R['radii']); perm=np.array(R['symmetry_perm'])-1
T=np.array([[0,1,0],[-1,0,0],[0,0,-1.]])
L_=lambda i,j: rr[i]+rr[j]
def angle(a,b,c):  # angle opposite side a, adjacent b,c (hyperbolic)
    return np.arccos(np.clip((np.cosh(b)*np.cosh(c)-np.cosh(a))/(np.sinh(b)*np.sinh(c)),-1,1))
def hyp_angles(t):
    i,j,k=t; a,b,c=L_(j,k),L_(i,k),L_(i,j)
    return angle(a,b,c),angle(b,c,a),angle(c,a,b)
# triangle -> nonagon
face_of=[]
for t in tri:
    hits=[f for f,face in enumerate(F) if set(t)<=set(face)]
    assert len(hits)==1; face_of.append(hits[0])
face_of=np.array(face_of)
# consistent orientation by BFS on the dual graph
edge_tris={}
for n,t in enumerate(tri):
    for a,b in [(t[0],t[1]),(t[1],t[2]),(t[2],t[0])]:
        edge_tris.setdefault(frozenset((a,b)),[]).append(n)
def oriented_edges(t): return [(t[0],t[1]),(t[1],t[2]),(t[2],t[0])]
def rotate_to(t,u,v):  # rotate oriented triangle so that it starts with u->v
    for s in range(3):
        tt=(t[s],t[(s+1)%3],t[(s+2)%3])
        if tt[0]==u and tt[1]==v: return tt
    return None
def mob(P):  # z -> (z+P)/(1+conj(P) z)
    return lambda z:(z+P)/(1+np.conj(P)*z)
def mobinv(P): return mob(-P)
def develop(root):
    import heapq
    o={root:tuple(tri[root])}; pos={}
    i,j,k=o[root]; ai,aj,ak=hyp_angles((i,j,k))
    pos[root]={i:0j, j:np.tanh(L_(i,j)/2)+0j, k:np.tanh(L_(i,k)/2)*np.exp(1j*ai)}
    parent={root:None}; order=[root]; heap=[]; cnt=0
    def push(n,d):
        nonlocal cnt
        for u,v in oriented_edges(o[n]):
            for m in edge_tris[frozenset((u,v))]:
                if m in o: continue
                w8=0 if face_of[m]==face_of[n] else 1
                cnt+=1; heapq.heappush(heap,(d+w8,cnt,m,n,u,v))
    push(root,0)
    while heap:
        d,_,m,n,u,v=heapq.heappop(heap)
        if m in o: continue
        t=tuple(tri[m]); tt=rotate_to(t,v,u) or rotate_to(t[::-1],v,u)
        o[m]=tt; w=tt[2]
        P,Q=pos[n][v],pos[n][u]
        th=np.angle(mobinv(P)(Q)); av=hyp_angles(tt)[0]
        zw=mob(P)(np.exp(1j*th)*np.tanh(L_(v,w)/2)*np.exp(1j*av))
        pos[m]={v:P,u:Q,w:zw}; parent[m]=(n,frozenset((u,v))); order.append(m)
        push(m,d)
    return o,pos,parent,order
best=None
for root in range(len(tri)):
    o,pos,parent,order=develop(root)
    pts=np.array([z for d in pos.values() for z in d.values()])
    # recentre: hyperbolic barycentre approx = Moebius sending mean to 0 (one step)
    c=pts.mean(); f=mobinv(c); m=np.abs(f(pts)).max()
    if best is None or m<best[0]: best=(m,root,c)
m,root,c=best
o,pos,parent,order=develop(root)
f=mobinv(c)
pos={n:{v:f(z) for v,z in d.items()} for n,d in pos.items()}
print('root',root,'max |z| of domain',m)
# sanity: angle sums and orientation consistency
sums=np.zeros(24)
for n,t in o.items():
    a=hyp_angles(t); 
    for v,x in zip(t,a): sums[v]+=x
print('max |angle sum - 2pi|',np.abs(sums-2*np.pi).max())
# boundary pairing: edges of domain that are glued (dual edges not in spanning tree)
tree_edges={p[1] for p in parent.values() if p}
cut=[e for e in edge_tris if e not in tree_edges]
print('glued edge pairs on boundary of fundamental polygon:',len(cut),' -> polygon with',2*len(cut),'sides')
json.dump(dict(root=int(root),orient={int(n):[int(x) for x in t] for n,t in o.items()},
               pos={int(n):{int(v):[z.real,z.imag] for v,z in d.items()} for n,d in pos.items()},
               face_of=face_of.tolist(),cut=[sorted(int(x) for x in e) for e in cut]),open(os.path.join(HERE, 'development.json'),'w'))
