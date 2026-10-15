"""The ribbon graph of Mizhaev's polyhedron P as a fat graph.
Vertices: the 24 vertices of P. Bands: the 36 edges; the two rims of each band carry the colours of the two nonagons
that meet along that edge. The cyclic order at each vertex comes from the orientation of the surface. Where the
planar drawing disagrees with that order the vertex is drawn 'face down' (open circle), and a band between a face-up
and a face-down vertex carries a half twist: its rim colours swap sides in the middle.
Mode 'flat' (default): every vertex face up, so no band is twisted; the layout minimises crossings, drawn as
over- and underpasses. Mode 'symmetric': tau acts as a quarter turn of the drawing; since tau reverses orientation and
a quarter turn does not, every second vertex of an orbit must then lie face down.
Usage: python visual/fig_ribbon.py OUTDIR [--mode flat|symmetric] [--restarts N] [--iters M] [--seed S] [--layout FILE]"""
import os, sys, json, argparse, numpy as np, networkx as nx, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(os.path.dirname(HERE), 'src')
ap = argparse.ArgumentParser(); ap.add_argument('out', nargs='?', default='.'); ap.add_argument('--samples', type=int, default=20000)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--restarts', type=int, default=8, help='flat mode: independent annealing runs')
ap.add_argument('--iters', type=int, default=4000, help='flat mode: annealing steps per run')
ap.add_argument('--layout', default=None, help='redraw a saved fig_ribbon_layout.json without searching')
ap.add_argument('--mode', choices=['flat', 'symmetric'], default='flat',
                help='flat: every vertex face up, no twists, fewest crossings; symmetric: C4-symmetric layout (forces twists)')
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8})
pal = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7']   # nonagon colours of the paper

# ---------------- ribbon graph from the oriented triangulation
R = json.load(open(os.path.join(SRC, 'results.json'))); D = json.load(open(os.path.join(SRC, 'development.json')))
TRI = [D['orient'][str(n)] for n in range(56)]                       # consistently oriented triangles of K
FACES = [[x - 1 for x in f] for f in R['faces']]
G = nx.Graph(); G.add_edges_from((f[i], f[(i + 1) % 9]) for f in FACES for i in range(9))
assert G.number_of_nodes() == 24 and G.number_of_edges() == 36 and all(d == 3 for _, d in G.degree())
face_of_tri = [next(k for k, f in enumerate(FACES) if set(t) <= set(f)) for t in TRI]
left = {}                                                            # nonagon on the left of the dart u->v
for t, k in zip(TRI, face_of_tri):
    for i in range(3):
        u, v = t[i], t[(i + 1) % 3]
        if G.has_edge(u, v): left[(u, v)] = k
assert len(left) == 72
def rotation(v):                                                     # neighbours of v, counter-clockwise on the surface
    nxt = {t[(t.index(v) + 1) % 3]: t[(t.index(v) + 2) % 3] for t in TRI if v in t}
    a0 = next(iter(nxt)); cyc = [a0]
    while nxt[cyc[-1]] != a0: cyc.append(nxt[cyc[-1]])
    return [u for u in cyc if G.has_edge(u, v)]
ROT = {v: rotation(v) for v in G}
perm = np.array(R['symmetry_perm']) - 1                              # tau on vertices
reps, orbit, power = [], {}, {}
for v in range(24):
    if v in orbit: continue
    reps.append(v); w = v
    for m in range(4): orbit[w], power[w] = len(reps) - 1, m; w = perm[w]
EDGES = list(G.edges())

# ---------------- C4-symmetric layout: one (radius, angle) per vertex orbit, tau = quarter turn of the drawing
def positions(params):
    pos = np.zeros((24, 2))
    for v in range(24):
        r, a = params[orbit[v]]; a = a - power[v] * np.pi / 2       # T(x, y) = (y, -x): a quarter turn clockwise
        pos[v] = r * np.cos(a), r * np.sin(a)
    return pos
def signs(pos):                                                      # +1: planar ccw order = surface order
    s = {}
    for v in range(24):
        nb = ROT[v]; ang = [np.arctan2(*(pos[u] - pos[v])[::-1]) for u in nb]
        order = [nb[i] for i in np.argsort(ang)]
        k = order.index(nb[0]); s[v] = 1 if order[k:] + order[:k] == nb else -1
    return s
def crossings(pos):
    P = np.array([[pos[u], pos[v]] for u, v in EDGES]); n = 0
    for i in range(len(EDGES)):
        for j in range(i + 1, len(EDGES)):
            if set(EDGES[i]) & set(EDGES[j]): continue
            a, b = P[i]; c, d = P[j]
            def side(p, q, r): return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
            if side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0: n += 1
    return n
def cost(params):
    pos = positions(params); s = signs(pos); tw = sum(s[u] != s[v] for u, v in EDGES)
    dmin = min(np.linalg.norm(pos[i] - pos[j]) for i in range(24) for j in range(i + 1, 24))
    return crossings(pos) + 0.5 * tw + 3.0 * (dmin < 0.08), tw
rng = np.random.default_rng(args.seed)
VP = np.array(R['vertices'], float)
def min_angle(pos):                                                  # smallest angle between two bands at a vertex
    m = np.pi
    for v in range(24):
        a = np.sort([np.arctan2(*(pos[u] - pos[v])[::-1]) for u in ROT[v]])
        m = min(m, np.min(np.diff(np.r_[a, a[0] + 2 * np.pi])))
    return m
if args.layout:
    pos = np.array(json.load(open(args.layout))['positions']); best = pos
elif args.mode == 'symmetric':
    start = np.array([[np.hypot(*VP[r, :2]) / 300, np.arctan2(VP[r, 1], VP[r, 0])] for r in reps])  # view down the axis
    best = start.copy(); bc, _ = cost(best)
    for it in range(args.samples):
        cand = best + rng.normal(scale=[0.15, 0.4], size=best.shape) if it % 4 else \
               np.c_[rng.uniform(0.15, 1.0, len(reps)), rng.uniform(-np.pi, np.pi, len(reps))]
        cand[:, 0] = np.clip(cand[:, 0], 0.12, 1.0); c, _ = cost(cand)
        if c < bc: best, bc = cand, c
    pos = positions(best)
else:
    def fcost(pos):
        s_ = signs(pos); down = sum(x < 0 for x in s_.values())
        dmin = min(np.linalg.norm(pos[i] - pos[j]) for i in range(24) for j in range(i + 1, 24))
        return crossings(pos) + 10 * down + 4 * max(0, 0.12 - dmin) / 0.12 + 2 * max(0, 0.45 - min_angle(pos)) / 0.45
    best_pos, bc = None, np.inf
    for restart in range(args.restarts):
        pos = nx.kamada_kawai_layout(G, pos=nx.spring_layout(G, seed=int(rng.integers(1 << 30))))
        pos = np.array([pos[v] for v in range(24)]); pos /= np.abs(pos).max(); c = fcost(pos); T0 = 1.0
        for it in range(args.iters):                                 # simulated annealing, one vertex at a time
            v = rng.integers(24); old = pos[v].copy(); pos[v] = np.clip(pos[v] + rng.normal(scale=0.08, size=2), -1, 1)
            cn = fcost(pos); T = T0 * (1 - it / args.iters) + 1e-3
            if cn <= c or rng.random() < np.exp((c - cn) / T): c = cn
            else: pos[v] = old
        if c < bc: best_pos, bc = pos.copy(), c
    pos = best_pos; best = pos
s = signs(pos); tw = sum(s[u] != s[v] for u, v in EDGES); cr = crossings(pos)
print(f'layout ({args.mode}): {cr} crossings, {tw} twisted bands, {sum(x < 0 for x in s.values())} face-down vertices')

# ---------------- checks: boundary cycles of the ribbon graph are the eight nonagons
def walk(u, v):
    cyc, (a, b) = [u], (u, v)
    while True:
        r = ROT[b]; w = r[(r.index(a) - 1) % 3]; cyc.append(b); a, b = b, w   # keeps the left face
        if (a, b) == (u, v): return cyc[:-1]
seen, bnd = set(), []
for d in sorted(left):
    if d in seen: continue
    c = walk(*d); bnd.append((left[d], c)); seen |= {(c[i], c[(i + 1) % len(c)]) for i in range(len(c))}
assert sorted(len(c) for _, c in bnd) == [9] * 8 and sorted(k for k, _ in bnd) == list(range(8)), 'boundary cycles must be the nonagons'
print('check: 8 boundary cycles = the 8 nonagons, genus', (2 - (24 - 36 + 8)) // 2)

# ---------------- drawing
fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=240); w = 0.011
Z = [2]
def rim(p, q, colour_left, colour_right):
    d = q - p; n = np.array([-d[1], d[0]]) / np.linalg.norm(d) * w
    ax.plot(*np.c_[p, q], color='white', lw=7.0, solid_capstyle='butt', zorder=Z[0])          # halo: gap at crossings
    ax.plot(*np.c_[p + n, q + n], color=colour_left, lw=1.5, solid_capstyle='butt', zorder=Z[0] + .1)
    ax.plot(*np.c_[p - n, q - n], color=colour_right, lw=1.5, solid_capstyle='butt', zorder=Z[0] + .1)
for u, v in EDGES:
    Z[0] += 0.2
    m = (pos[u] + pos[v]) / 2
    # near u: surface-left of u->v is drawing-left if u is face up; near v the same with v's sign
    L, Rr = pal[left[(u, v)]], pal[left[(v, u)]]
    rim(pos[u], m, *((L, Rr) if s[u] > 0 else (Rr, L)))
    rim(m, pos[v], *((L, Rr) if s[v] > 0 else (Rr, L)))
    if s[u] != s[v]: ax.plot(*m, marker='x', color='#1b2733', ms=3.5, mew=.8, zorder=3)
for v in range(24):
    ax.plot(*pos[v], 'o', ms=7.5, mfc='#1b2733' if s[v] > 0 else 'white', mec='#1b2733', mew=.9, zorder=100)
    a = np.sort([np.arctan2(*(pos[u] - pos[v])[::-1]) for u in ROT[v]]); gaps = np.diff(np.r_[a, a[0] + 2 * np.pi])
    k = int(np.argmax(gaps)); phi = a[k] + gaps[k] / 2                  # label in the widest free sector
    ax.text(*(pos[v] + 0.065 * np.array([np.cos(phi), np.sin(phi)])), str(v + 1), ha='center', va='center', fontsize=6.5, zorder=101)
ax.set_aspect('equal'); ax.axis('off')
fig.savefig(os.path.join(args.out, 'fig_ribbon.pdf'), bbox_inches='tight', pad_inches=.02)
fig.savefig(os.path.join(args.out, 'fig_ribbon.png'), bbox_inches='tight', pad_inches=.02)
json.dump(dict(mode=args.mode, seed=args.seed, positions=pos.tolist(), crossings=int(cr), twists=int(tw), min_angle_deg=float(np.degrees(min_angle(pos))), rotation={str(v + 1): [u + 1 for u in ROT[v]] for v in range(24)}),
          open(os.path.join(args.out, 'fig_ribbon_layout.json'), 'w'), indent=1)
print('written', os.path.join(args.out, 'fig_ribbon.pdf'))
