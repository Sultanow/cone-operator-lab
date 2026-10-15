"""Figure: horizontal cross-sections of the solid bounded by P. From top to bottom: one wall, two walls, a ring
around the symmetry axis, two walls (turned by 90 degrees), one wall. Usage: python visual/fig_slices.py OUTDIR"""
import os, sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.path import Path; from matplotlib.patches import PathPatch
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(os.path.dirname(HERE), 'src')
OUT = sys.argv[1] if len(sys.argv) > 1 else '.'; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8.5})
R = json.load(open(os.path.join(SRC, 'results.json'))); V = np.array(R['vertices'], float); TRI = np.array(R['triangles']) - 1
def loops_at(c):
    seg = {}
    for t, (a, b, d) in enumerate(TRI):
        cr = [frozenset((u, v)) for u, v in ((a, b), (b, d), (d, a)) if (V[u, 2] - c) * (V[v, 2] - c) < 0]
        if cr: seg[t] = cr
    et = {}
    for t, cr in seg.items():
        for e in cr: et.setdefault(e, []).append(t)
    def pt(e):
        u, v = tuple(e); s = (c - V[u, 2]) / (V[v, 2] - V[u, 2]); return (V[u] + s * (V[v] - V[u]))[:2]
    out, used = [], set()
    for t0 in seg:
        if t0 in used: continue
        pts = []; used.add(t0); e = seg[t0][1]; t = t0
        while True:
            pts.append(pt(e)); nt = [x for x in et[e] if x != t][0]
            if nt == t0: break
            used.add(nt); e = [x for x in seg[nt] if x != e][0]; t = nt
        out.append(np.array(pts))
    return out
H = [(190, 'one wall'), (60, 'two walls'), (0, 'a ring around the axis'), (-60, 'two walls'), (-190, 'one wall')]
fig, axs = plt.subplots(1, 5, figsize=(7.0, 1.95), dpi=240)
for ax, (z, lab) in zip(axs, H):
    L = loops_at(z + 0.37)
    def area(l): return 0.5 * np.sum(l[:, 0] * np.roll(l[:, 1], -1) - l[:, 1] * np.roll(l[:, 0], -1))
    def inside(p, l): return Path(l).contains_point(p)
    depth = [sum(inside(L[i][0], L[j]) for j in range(len(L)) if j != i) for i in range(len(L))]
    L = [l if (area(l) > 0) == (d % 2 == 0) else l[::-1] for l, d in zip(L, depth)]   # holes run the other way
    verts = np.vstack([np.vstack([l, l[:1]]) for l in L]); codes = []
    for l in L: codes += [Path.MOVETO] + [Path.LINETO] * (len(l) - 1) + [Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor='#4E79A7', alpha=.55, edgecolor='#1b2733', lw=.6))   # even-odd fill
    ax.plot(0, 0, marker='+', color='#E15759', ms=6, mew=1.2)
    ax.set_xlim(-310, 310); ax.set_ylim(-310, 310); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'$z={z}$: {lab}', fontsize=7.5)
fig.subplots_adjust(wspace=0.05, left=0, right=1, top=.86, bottom=0)
fig.savefig(os.path.join(OUT, 'fig_slices.pdf'), bbox_inches='tight', pad_inches=.02)
fig.savefig(os.path.join(OUT, 'fig_slices.png'), bbox_inches='tight', pad_inches=.02); print('written fig_slices.pdf')
