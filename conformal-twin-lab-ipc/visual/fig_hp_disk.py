"""Figure 4 of the paper: the canonical twin h_P (uniformised polyhedron metric, level 3) developed into the
Poincare disk, with the 24 cone points of P marked by their cone angle. Also writes hp_disk.json for
visual/mizhaev_visual.wl.  Usage: python visual/fig_hp_disk.py OUTDIR"""
import numpy as np, json, os, sys, heapq, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(os.path.dirname(HERE), 'src')
sys.path.insert(0, SRC); sys.path.insert(0, os.path.join(SRC, 'wpc'))
OUT = sys.argv[1] if len(sys.argv) > 1 else '.'; os.makedirs(OUT, exist_ok=True)
from fine_mesh import build
from uniformize import uniformize
from holonomy import place, angle
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8.5})
R = json.load(open(os.path.join(SRC, 'results.json'))); D = json.load(open(os.path.join(SRC, 'development.json'))); VP = np.array(R['vertices'], float)
theta = np.degrees(2 * np.pi - np.array(R['original_curvature']))
LEV = 3; M = build(LEV, SRC); U = uniformize(M, verbose=False); F = [tuple(map(int, f)) for f in M['F']]
Ld = {frozenset(map(int, e)): l for e, l in zip(M['edges'], U['L'])}; L = lambda i, j: Ld[frozenset((i, j))]
nonagon = np.array(D['face_of'])[M['face_tri']]
ef = {}
for f, (a, b, c) in enumerate(F):
    for u, v in ((a, b), (b, c), (c, a)): ef.setdefault(frozenset((u, v)), []).append(f)
def develop(root):
    """placement directly in the Poincare disk with Moebius maps (numerically stable for deep trees)"""
    def put(zv, zu, Lvw, alpha):
        g = lambda z: (z - zv) / (1 - np.conj(zv) * z); phi = np.angle(g(zu))
        w = np.tanh(Lvw / 2) * np.exp(1j * (phi + alpha)); return (w + zv) / (1 + np.conj(zv) * w)
    a, b, c = F[root]; pos = {root: {a: 0j, b: np.tanh(L(a, b) / 2) + 0j}}
    pos[root][c] = put(pos[root][a], pos[root][b], L(a, c), angle(L(b, c), L(a, b), L(a, c)))
    heap = []; cnt = 0
    def push(f, d):
        nonlocal cnt
        a, b, c = F[f]
        for u, v in ((a, b), (b, c), (c, a)):
            for g in ef[frozenset((u, v))]:
                if g != f and g not in pos: cnt += 1; heapq.heappush(heap, (d + (nonagon[g] != nonagon[f]), cnt, g, f, u, v))
    push(root, 0)
    while heap:
        d, _, g, f, u, v = heapq.heappop(heap)
        if g in pos: continue
        w = [x for x in F[g] if x not in (u, v)][0]
        pos[g] = {v: pos[f][v], u: pos[f][u], w: put(pos[f][v], pos[f][u], L(v, w), angle(L(u, w), L(u, v), L(v, w)))}
        push(g, d)
    return pos
def to_disk(z): return z
best = None
for root in [int(np.where(M['face_tri'] == t)[0][0]) for t in range(56)]:
    pos = develop(root); Z = np.array([to_disk(p) for d in pos.values() for p in d.values()])
    c = Z.mean(); W = (Z - c) / (1 - np.conj(c) * Z); m = np.abs(W).max()
    if best is None or m < best[0]: best = (m, root, c)
m, root, c = best; pos = develop(root); mob = lambda z: (z - c) / (1 - np.conj(c) * z)
print('root face', root, 'max |z|', round(m, 4))
pal = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7']
polys, cols = [], []
for f, d in pos.items():
    z = [mob(to_disk(d[v])) for v in F[f]]; polys.append([(q.real, q.imag) for q in z]); cols.append(pal[nonagon[f]])
# edges of P = fine edges lying on an original polyhedron edge
orig = [(f[k] - 1, f[(k + 1) % 9] - 1) for f in R['faces'] for k in range(9)]
XP = M['XP']
def on_orig(a, b):
    for p, q in orig:
        d = VP[q] - VP[p]
        ok = True
        for x in (a, b):
            t = np.dot(XP[x] - VP[p], d) / np.dot(d, d)
            if not (-1e-9 <= t <= 1 + 1e-9 and np.linalg.norm(XP[x] - (VP[p] + t * d)) < 1e-6): ok = False; break
        if ok: return True
    return False
segs = []
for f, d in pos.items():
    a, b, cc = F[f]
    for u, v in ((a, b), (b, cc), (cc, a)):
        if on_orig(u, v):
            zu, zv = mob(to_disk(d[u])), mob(to_disk(d[v])); segs.append([(zu.real, zu.imag), (zv.real, zv.imag)])
marked = {int(np.argmin(np.linalg.norm(XP - VP[i], axis=1))): i for i in range(24)}
pts = {}
for f, d in pos.items():
    for v, p in d.items():
        if v in marked: z = mob(to_disk(p)); pts[(round(z.real, 7), round(z.imag, 7))] = theta[marked[v]]
fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=220)
ax.add_patch(plt.Circle((0, 0), 1, fc='#f4f6f8', ec='#2b3a47', lw=1.0, zorder=0))
ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors=cols, linewidths=.3, alpha=.6, zorder=1))
ax.add_collection(LineCollection(segs, colors='#1b2733', linewidths=.7, zorder=2))
xy = np.array(list(pts.keys())); th = np.array(list(pts.values()))
sc = ax.scatter(xy[:, 0], xy[:, 1], c=th, cmap='coolwarm', vmin=0, vmax=720, s=22, edgecolors='#1b2733', linewidths=.5, zorder=3)
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02); cb.set_label('cone angle of the vertex on $P$ (degrees)'); cb.set_ticks([0, 95.5, 360, 682.6])
ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02); ax.set_aspect('equal'); ax.axis('off')
fig.savefig(os.path.join(OUT, 'fig_hp_disk.pdf'), bbox_inches='tight', pad_inches=.02); fig.savefig(os.path.join(OUT, 'fig_hp_disk.png'), bbox_inches='tight', pad_inches=.02)
tris = [dict(nonagon=int(nonagon[f]) + 1, xy=[[float(mob(d[v]).real), float(mob(d[v]).imag)] for v in F[f]]) for f, d in pos.items()]
json.dump(dict(note='canonical twin h_P (level 3) in the Poincare disk', triangles=tris,
               edges=[[list(map(float, a)), list(map(float, b))] for a, b in segs],
               cone_points=[dict(xy=[float(k[0]), float(k[1])], angle=float(v)) for k, v in pts.items()]),
          open(os.path.join(OUT, 'hp_disk.json'), 'w'))
print('marked point copies drawn:', len(pts), ' edge segments of P:', len(segs))
