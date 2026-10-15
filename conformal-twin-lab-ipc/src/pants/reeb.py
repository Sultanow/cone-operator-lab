"""Pants decomposition from the Reeb graph of a slightly tilted height function on the PL surface P (triangulation K).
Levels are sampled between consecutive vertex heights; loops of consecutive levels are joined when they cross a common
edge. Removing the leaf chains (they end in a maximum or minimum and bound discs) and smoothing degree-2 nodes leaves a
trivalent graph with 2g-2 = 4 nodes (the pants) and 3g-3 = 6 edges (the cuffs). Each cuff is a closed sequence of the
56 triangles of K, ready for the holonomy length computation."""
import json, numpy as np
import os
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(SRC, 'results.json'))); V = np.array(R['vertices'], float)
D = json.load(open(os.path.join(SRC, 'development.json'))); TRI = np.array([D['orient'][str(n)] for n in range(56)])   # consistently oriented
def build(tilt=(0.0031, 0.0017)):
    h = V[:, 2] + tilt[0] * V[:, 0] + tilt[1] * V[:, 1]
    assert len(set(np.round(h, 9))) == 24, 'heights must be distinct'
    order = np.sort(h); levels = (order[:-1] + order[1:]) / 2
    def loops_at(c):
        seg = {}
        for t, (a, b, d) in enumerate(TRI):
            cr = [frozenset((u, v)) for u, v in ((a, b), (b, d), (d, a)) if (h[u] - c) * (h[v] - c) < 0]
            if cr: seg[t] = cr
        et = {}
        for t, cr in seg.items():
            for e in cr: et.setdefault(e, []).append(t)
        out, used = [], set()
        for t0 in seg:
            if t0 in used: continue
            tris, edges = [t0], set(); used.add(t0); e = seg[t0][1]; t = t0
            while True:
                edges.add(e); nt = [x for x in et[e] if x != t][0]
                if nt == t0: break
                tris.append(nt); used.add(nt); e = [x for x in seg[nt] if x != e][0]; t = nt
            out.append(dict(tris=tris, edges=edges, level=c))
        return out
    L = [loops_at(c) for c in levels]
    nodes = [(i, k) for i in range(len(L)) for k in range(len(L[i]))]
    adj = {n: set() for n in nodes}
    for i in range(len(L) - 1):
        for a, la in enumerate(L[i]):
            for b, lb in enumerate(L[i + 1]):
                if la['edges'] & lb['edges']: adj[(i, a)].add((i + 1, b)); adj[(i + 1, b)].add((i, a))
    # critical structure: a node with != 1 neighbour above or below sits next to a critical point
    return h, L, adj
def pants_decomposition(tilt=(0.0031, 0.0017)):
    h, L, adj = build(tilt)
    G = {n: set(s) for n, s in adj.items()}
    changed = True
    while changed:                                     # strip leaf chains (towards maxima / minima)
        changed = False
        for n in list(G):
            if len(G[n]) <= 1:
                for m in G[n]: G[m].discard(n)
                del G[n]; changed = True
    # remaining graph: chains of degree-2 loop nodes between junctions; a junction is where loops merge/split
    # (a loop node with 3 neighbours). Collapse chains into edges of the trivalent graph.
    junctions = [n for n in G if len(G[n]) != 2]
    edges, seen = [], set()
    for j in junctions:
        for m in G[j]:
            if (j, m) in seen: continue
            chain = [j, m]; prev, cur = j, m
            while len(G[cur]) == 2:
                nxt = [x for x in G[cur] if x != prev][0]; chain.append(nxt); prev, cur = cur, nxt
            seen.add((chain[0], chain[1])); seen.add((chain[-1], chain[-2]))
            edges.append(chain)
    return h, L, G, junctions, edges
if __name__ == '__main__':
    h, L, adj = build()
    print('loops per level:', [len(l) for l in L])
    chi = 0
    for v in range(24):   # PL index from the link: chi = sum (1 - sign changes / 2)
        nb = [t for t in TRI if v in t]
        link = {}
        for t in nb:
            i = list(t).index(v); a, b = t[(i + 1) % 3], t[(i + 2) % 3]; link[a] = b
        a0 = next(iter(link)); cyc = [a0]
        while link[cyc[-1]] != a0: cyc.append(link[cyc[-1]])
        s = np.sign(h[cyc] - h[v]); ch = int(np.sum(s != np.roll(s, 1)))
        chi += 1 - ch / 2
    print('Euler characteristic from PL critical points:', chi)

def trivalent(tilt=(0.0031, 0.0017)):
    """events between consecutive levels: continuation (1-1), saddle (2-1 or 1-2), extremum (1-0 or 0-1).
    Segments = maximal chains of loops joined by continuations. Returns pants (saddles) and cuffs (segments)."""
    h, L, adj = build(tilt)
    parent = {(i, k): (i, k) for i in range(len(L)) for k in range(len(L[i]))}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    events = []                                        # (kind, below loops, above loops)
    nb = lambda n, j: [m for m in adj[n] if m[0] == j]
    for i in range(-1, len(L)):
        below = [(i, k) for k in range(len(L[i]))] if i >= 0 else []
        above = [(i + 1, k) for k in range(len(L[i + 1]))] if i + 1 < len(L) else []
        seenb, seena = set(), set()
        for start in below + above:                     # components of the bipartite graph between the two levels
            if start in seenb or start in seena: continue
            comp, stack = set(), [start]
            while stack:
                x = stack.pop()
                if x in comp: continue
                comp.add(x); stack += [m for m in adj[x] if m[0] in (i, i + 1)]
            B = [x for x in comp if x[0] == i]; A = [x for x in comp if x[0] == i + 1]
            seenb |= set(B); seena |= set(A)
            if len(B) == 1 and len(A) == 1: parent[find(B[0])] = find(A[0])
            else: events.append((B, A))
    seg = {}
    for n in parent: seg.setdefault(find(n), []).append(n)
    segs = {r: sorted(v) for r, v in seg.items()}
    # graph: event nodes, segment edges
    inc = {r: [] for r in segs}
    for e, (B, A) in enumerate(events):
        for x in B + A: inc[find(x)].append(e)
    deg = {e: sum(e in v for v in inc.values()) for e in range(len(events))}
    # strip segments ending in an extremum (event with a single incident segment)
    alive = dict(segs); ev_alive = set(range(len(events)))
    changed = True
    while changed:
        changed = False
        for e in list(ev_alive):
            s_inc = [r for r in alive if e in inc[r]]
            if len(s_inc) <= 1:
                for r in s_inc: del alive[r]
                ev_alive.discard(e); changed = True
    # smooth events of degree 2 (merge their two segments into one cuff)
    cuffs = {r: [r] for r in alive}; inc2 = {r: [e for e in inc[r] if e in ev_alive] for r in alive}
    for e in list(ev_alive):
        s_inc = [r for r in cuffs if any(e in inc2[x] for x in cuffs[r])]
        if len(s_inc) == 2:
            a, b = s_inc; cuffs[a] += cuffs.pop(b); ev_alive.discard(e)
    pants = {e: [r for r in cuffs if any(e in inc2[x] for x in cuffs[r])] for e in ev_alive}
    return h, L, segs, cuffs, pants

def trivalent2(tilt=(0.0031, 0.0017)):
    """same as trivalent, but as a multigraph: every segment has a lower and an upper event."""
    h, L, segs, _, _ = trivalent(tilt)
    _, _, adj = build(tilt)
    # recompute events with their lower/upper loops
    events = []
    for i in range(-1, len(L)):
        below = [(i, k) for k in range(len(L[i]))] if i >= 0 else []
        above = [(i + 1, k) for k in range(len(L[i + 1]))] if i + 1 < len(L) else []
        done = set()
        for start in below + above:
            if start in done: continue
            comp, stack = set(), [start]
            while stack:
                x = stack.pop()
                if x in comp: continue
                comp.add(x); stack += [m for m in adj[x] if m[0] in (i, i + 1)]
            done |= comp
            B = [x for x in comp if x[0] == i]; A = [x for x in comp if x[0] == i + 1]
            if not (len(B) == 1 and len(A) == 1): events.append((B, A))
    seg_of = {n: r for r, v in segs.items() for n in v}
    ends = {r: [None, None] for r in segs}               # [lower event, upper event]
    for e, (B, A) in enumerate(events):
        for x in B: ends[seg_of[x]][1] = e                # event above the segment's top loop
        for x in A: ends[seg_of[x]][0] = e
    edges = {r: tuple(v) for r, v in ends.items()}       # multigraph edge r: (event, event)
    members = {r: [r] for r in edges}
    def degree(e): return sum((a == e) + (b == e) for a, b in edges.values())
    changed = True
    while changed:
        changed = False
        for e in range(len(events)):
            d = degree(e)
            if d == 1:                                    # leaf: drop its segment
                r = [r for r, (a, b) in edges.items() if e in (a, b)][0]; del edges[r]; changed = True
            elif d == 2:
                rs = [r for r, (a, b) in edges.items() if e in (a, b)]
                if len(rs) == 2:                          # two different segments meet: concatenate
                    r1, r2 = rs; o1 = [x for x in edges[r1] if x != e][0] if edges[r1].count(e) == 1 else e
                    o2 = [x for x in edges[r2] if x != e][0] if edges[r2].count(e) == 1 else e
                    edges[r1] = (o1, o2); members[r1] += members[r2]; del edges[r2]; changed = True
    nodes = sorted({x for ab in edges.values() for x in ab})
    return h, L, segs, events, edges, members, nodes
