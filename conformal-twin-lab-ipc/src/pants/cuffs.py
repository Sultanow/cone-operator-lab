"""Section "Pants": lengths of the six cuffs of the Reeb pants decomposition in h_cp and h_P (levels 1-3), and of their images
under the symmetry tau, by holonomy along the realised dual loops.  Usage: python src/pants/cuffs.py OUTDIR"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC); sys.path.insert(0, os.path.join(SRC, 'wpc')); sys.path.insert(0, HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else '.'; os.makedirs(OUT, exist_ok=True)
from fine_mesh import build
from holonomy import Developed, length_of
from loops import realise
from uniformize import uniformize
from reeb import trivalent2
R = json.load(open(f'{SRC}/results.json')); perm = np.array(R['symmetry_perm']) - 1
TRIs = [frozenset(np.array(t) - 1) for t in R['triangles']]; tri_of = {s: k for k, s in enumerate(TRIs)}
tau_tri = [tri_of[frozenset(perm[list(s)])] for s in TRIs]
h, L, segs, events, edges, members, nodes = trivalent2()
cuffs = []
for r in edges:
    loops = sorted(n for m in members[r] for n in segs[m]); n = loops[len(loops) // 2]
    cuffs.append(L[n[0]][n[1]]['tris'])
pants = {p: [k for k, (a, b) in enumerate(edges.values()) for _ in range((a == p) + (b == p))] for p in nodes}
def lengths(M, Lh, loops):
    Ld = {frozenset(map(int, e)): l for e, l in zip(M['edges'], Lh)}
    dev = Developed([tuple(map(int, f)) for f in M['F']], lambda i, j: Ld[frozenset((i, j))])
    fadj = {f: [] for f in range(len(M['F']))}
    for fs in dev.edge_faces.values(): fadj[fs[0]].append(fs[1]); fadj[fs[1]].append(fs[0])
    return np.array([length_of(dev.holonomy(realise(lp, M, fadj))) for lp in loops])
images = [[tau_tri[t] for t in c] for c in cuffs]
res = {}
M0 = build(0, SRC); res['cp'] = (lengths(M0, M0['lenH'], cuffs), lengths(M0, M0['lenH'], images))
for lev in (1, 2, 3):
    M = build(lev, SRC); U = uniformize(M, verbose=False); res[f'P{lev}'] = (lengths(M, U['L'], cuffs), lengths(M, U['L'], images))
print('pants -> cuffs:', pants)
print('cuff  ' + ''.join(f'{k:>10s}' for k in res))
for i in range(6): print(f'  {i}   ' + ''.join(f'{res[k][0][i]:10.5f}' for k in res))
print('tau-images of the cuffs (same order):')
for i in range(6): print(f'  t({i}) ' + ''.join(f'{res[k][1][i]:10.5f}' for k in res))
json.dump(dict(pants={str(p): v for p, v in pants.items()}, cuffs=[[int(t) for t in c] for c in cuffs],
               lengths={k: [v[0].tolist(), v[1].tolist()] for k, v in res.items()}), open(os.path.join(OUT, 'cuff_lengths.json'), 'w'), indent=1)
