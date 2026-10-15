#!/usr/bin/env python3
"""Exact mod-prime homology and C4 action certificate for Mizhaev triangulation.

Only reads the immutable results.json from the prior v1 experiment.
No numerical spinors or Dirac solutions are computed here.
"""
from pathlib import Path
import json
import numpy as np

BASE = Path(__file__).resolve().parent            # src/
R = json.loads((BASE/'results.json').read_text())
V = len(R['vertices'])
E = [tuple(int(x)-1 for x in e) for e in R['edges']]
F = [tuple(int(x)-1 for x in f) for f in R['triangles']]
perm = np.asarray(R['symmetry_perm'],dtype=int)-1
assert V==24 and len(E)==84 and len(F)==56
edge_idx={frozenset(e):i for i,e in enumerate(E)}
tri_idx={frozenset(f):i for i,f in enumerate(F)}
assert len(edge_idx)==len(E) and len(tri_idx)==len(F)

# Arbitrary but fixed orientations for simplicial 1- and 2-chains.
B1=np.zeros((V,len(E)),dtype=int)
for j,(a,b) in enumerate(E): B1[b,j]=1;B1[a,j]=-1
B2=np.zeros((len(E),len(F)),dtype=int)
for j,(a,b,c) in enumerate(F):
    for u,v in [(a,b),(b,c),(c,a)]:
        i=edge_idx[frozenset((u,v))]
        B2[i,j]=1 if E[i]==(u,v) else -1
assert not np.any(B1@B2)

def rank_mod_p(M,p=101):
    A=np.array(M,dtype=np.int64)%p
    n,m=A.shape; r=0
    for c in range(m):
        idx=np.flatnonzero(A[r:,c]);
        if len(idx)==0:continue
        z=r+int(idx[0]);A[[r,z],:]=A[[z,r],:]
        A[r,:]=(A[r,:]*pow(int(A[r,c]),-1,p))%p
        for k in range(n):
            if k!=r and A[k,c]:A[k,:]=(A[k,:]-A[k,c]*A[r,:])%p
        r+=1
        if r==n:break
    return r

rank_b1,rank_b2=rank_mod_p(B1),rank_mod_p(B2)
betti=(V-rank_b1,len(E)-rank_b1-rank_b2,len(F)-rank_b2)
assert betti==(1,6,1), (betti,rank_b1,rank_b2)

# Signed trace of the simplicial chain actions T^j.
def parity_of_permutation(arr):
    return (-1)**sum(arr[i]>arr[k] for i in range(len(arr)) for k in range(i+1,len(arr)))
def traces(q):
    tr0=int(sum(q[i]==i for i in range(V)))
    tr1=0; tr2=0; reversed_fixed_edges=[]
    for a,b in E:
        if frozenset((q[a],q[b]))==frozenset((a,b)):
            sign=1 if (q[a],q[b])==(a,b) else -1
            tr1+=sign
            if sign<0:reversed_fixed_edges.append((a+1,b+1))
    for a,b,c in F:
        x=(int(q[a]),int(q[b]),int(q[c]))
        if frozenset(x)==frozenset((a,b,c)):
            pos=[(a,b,c).index(y) for y in x]
            tr2+=parity_of_permutation(pos)
    return (tr0,tr1,tr2),reversed_fixed_edges

rows=[]
q=np.arange(V)
for j in range(4):
    chain,rev=traces(q)
    h0=1; h2=(-1)**j
    h1=h0+h2-(chain[0]-chain[1]+chain[2])
    rows.append(dict(power=j,chain_traces=chain,H1_trace=h1,reversed_fixed_edges=rev))
    q=perm[q]
assert [r['H1_trace'] for r in rows]==[6,0,-2,0], rows
assert sorted(rows[2]['reversed_fixed_edges'])==[(5,8),(18,22),(19,23),(20,24)]
# Character decomposition: H1 ~= +1 (+1 copy) | -1 (1 copy) | plane_rotation (2 copies).
rep_H1={'trivial_real_1D':1,'sign_real_1D':1,'quarter_turn_real_2D':2}
rep_R3={'trivial_real_1D':0,'sign_real_1D':1,'quarter_turn_real_2D':1}
# Endomorphisms of the quarter-turn real representation form C over R (dim_R 2).
period_parameter_dimension=1*1 + 2*1*2
assert period_parameter_dimension==5
result=dict(chain_ranks=[rank_b1,rank_b2],betti_numbers=list(betti),
    traces=rows,H1_decomposition=rep_H1,R3_decomposition=rep_R3,
    general_real_period_conditions=2*3*3,
    equivariant_real_period_parameters=period_parameter_dimension,
    note='5 is the dimension of possible equivariant period maps, NOT a proof that a spinor reconstruction exists or closes.')
import sys
out=(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent) / 'symmetry_period_certificate.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
