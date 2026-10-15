"""Usage: python src/ribbon.py OUTDIR.  The ribbon graph of P: the edge graph (24 vertices, 36 edges, trivalent) with the cyclic order of the three
edges at each vertex induced by the orientation of the surface. Its boundary cycles must be the eight nonagons."""
import os, sys, json, numpy as np
SRC = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else '.'; os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(SRC, 'results.json'))); D = json.load(open(os.path.join(SRC, 'development.json')))
TRI = [D['orient'][str(n)] for n in range(56)]                       # consistently oriented triangles of K
FACES = [[x - 1 for x in f] for f in R['faces']]
Pedges = {frozenset((f[i], f[(i + 1) % 9])) for f in FACES for i in range(9)}
assert len(Pedges) == 36
def rotation(v):
    nxt = {}
    for t in TRI:
        if v in t:
            i = t.index(v); nxt[t[(i + 1) % 3]] = t[(i + 2) % 3]
    a0 = next(iter(nxt)); cyc = [a0]
    while nxt[cyc[-1]] != a0: cyc.append(nxt[cyc[-1]])
    nb = [u for u in cyc if frozenset((u, v)) in Pedges]
    assert len(nb) == 3; k = nb.index(min(nb)); return nb[k:] + nb[:k]      # start at the smallest label
rot = {v: rotation(v) for v in range(24)}
# boundary cycles: from dart u->v continue with v->w, w = the neighbour of v following u in the cyclic order
def walk(u, v):
    cyc = [u]; a, b = u, v
    while True:
        r = rot[b]; w = r[(r.index(a) + 1) % 3]; cyc.append(b); a, b = b, w
        if (a, b) == (u, v): return cyc[:-1]
darts = {(u, v) for e in Pedges for u, v in (tuple(e), tuple(e)[::-1])}; seen = set(); bnd = []
for d in sorted(darts):
    if d in seen: continue
    c = walk(*d); bnd.append(c)
    for i in range(len(c)): seen.add((c[i], c[(i + 1) % len(c)]))
canon = lambda c: min(tuple(c[i:] + c[:i]) for i in range(len(c)))
faces_canon = {canon(f) for f in FACES} | {canon(f[::-1]) for f in FACES}
print('boundary cycles:', len(bnd), 'lengths', sorted(len(c) for c in bnd),
      '| all are nonagons of P:', all(canon(c) in faces_canon for c in bnd), '| genus', (2 - (24 - 36 + len(bnd))) // 2)
perm = np.array(R['symmetry_perm']) - 1
cyc_ok = all([perm[x] for x in rot[v]] in [rot[perm[v]][i:] + rot[perm[v]][:i] for i in range(3)] or
             [perm[x] for x in rot[v]][::-1] in [rot[perm[v]][i:] + rot[perm[v]][:i] for i in range(3)] for v in range(24))
rev = sum([perm[x] for x in rot[v]] not in [rot[perm[v]][i:] + rot[perm[v]][:i] for i in range(3)] for v in range(24))
print('tau maps the ribbon structure to itself:', cyc_ok, '| cyclic orders reversed at', rev, 'of 24 vertices (tau reverses orientation)')
json.dump({str(v + 1): [x + 1 for x in rot[v]] for v in range(24)}, open(os.path.join(OUT, 'ribbon.json'), 'w'))
for v in range(24): print(v + 1, [x + 1 for x in rot[v]], end='; ' if v % 6 != 5 else '\n')
