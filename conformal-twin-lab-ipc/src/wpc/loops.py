"""Closed loops of coarse triangles (the 56 triangles of K) and their realisation as fine dual paths."""
import json
from collections import deque
def coarse_loops(path='.'):
    D = json.load(open(f'{path}/development.json')); F0 = [tuple(D['orient'][str(n)]) for n in range(56)]
    ef = {}
    for f, (a, b, c) in enumerate(F0):
        for u, v in ((a, b), (b, c), (c, a)): ef.setdefault(frozenset((u, v)), []).append(f)
    adj = {f: [] for f in range(56)}
    for fs in ef.values(): adj[fs[0]].append(fs[1]); adj[fs[1]].append(fs[0])
    par = {0: None}; q = deque([0])
    while q:
        f = q.popleft()
        for g in adj[f]:
            if g not in par: par[g] = f; q.append(g)
    def up(f):
        p = [f]
        while par[p[-1]] is not None: p.append(par[p[-1]])
        return p
    loops = []
    for fs in ef.values():
        a, b = fs
        if par.get(b) == a or par.get(a) == b: continue
        pa, pb = up(a), up(b)
        loops.append(list(reversed(pa)) + pb[:-1])     # root .. a | b .. (child of root)
    return loops, adj
def realise(loop, M, fine_adj):
    ft = M['face_tri']; inside = {}
    for f, t in enumerate(ft): inside.setdefault(int(t), []).append(f)
    def bfs(start, ok, goal):
        prev = {start: None}; q = deque([start])
        while q:
            f = q.popleft()
            if goal(f):
                p = [f]
                while prev[p[-1]] is not None: p.append(prev[p[-1]])
                return p[::-1]
            for g in fine_adj[f]:
                if g not in prev and ok(g): prev[g] = f; q.append(g)
        raise RuntimeError('no path')
    start = inside[loop[0]][0]; cur = start; path = [cur]
    for t_next in loop[1:] + [loop[0]]:
        t_cur = int(ft[cur])
        seg = bfs(cur, lambda g: ft[g] == t_cur, lambda g: any(ft[h] == t_next for h in fine_adj[g]))
        path += seg[1:]; cur = path[-1]
        nxt = [h for h in fine_adj[cur] if ft[h] == t_next][0]; path.append(nxt); cur = nxt
    seg = bfs(cur, lambda g: ft[g] == loop[0], lambda g: g == start); path += seg[1:]
    return path[:-1]
