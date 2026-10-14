"""Klein-linear refinement of the circle-packing triangulation K.

Each hyperbolic triangle of (S, h_cp) is subdivided into n^2 geodesic triangles
(n = 2^level) by radially projecting the barycentric grid of its hyperboloid
vertices.  The construction is equivariant under the order-4 symmetry tau, so the
fine mesh carries
  * fine vertex positions on Mizhaev's polyhedron P (same barycentric grid),
  * exact hyperbolic edge lengths of the fine geodesic triangulation,
  * the vertex permutation induced by tau.
"""
import json, numpy as np

def build(level, path='.'):
    R = json.load(open(f'{path}/results.json')); D = json.load(open(f'{path}/development.json'))
    V0 = np.array(R['vertices'], float)
    orient = {int(k): tuple(v) for k, v in D['orient'].items()}
    pos = {int(n): {int(v): complex(*z) for v, z in d.items()} for n, d in D['pos'].items()}
    perm0 = np.array(R['symmetry_perm']) - 1
    n = 2 ** level
    def lift(z):
        s = abs(z) ** 2
        return np.array([2 * z.real, 2 * z.imag, 1 + s]) / (1 - s)
    keys, verts_P, verts_H = {}, [], []
    def key_of(t, i, j):
        a, b, c = orient[t]; lam = (n - i - j, i, j); corners = (a, b, c)
        nz = [(corners[m], lam[m]) for m in range(3) if lam[m] > 0]
        if len(nz) == 1: return ('v', nz[0][0])
        if len(nz) == 2:
            (p, lp), (q, lq) = sorted(nz)
            return ('e', p, q, lq)                      # fraction lq/n from p towards q
        return ('f', frozenset(zip(corners, lam)))
    faces, face_tri, grid = [], [], {}
    for t in range(56):
        a, b, c = orient[t]
        PH = [lift(pos[t][v]) for v in (a, b, c)]
        idx = {}
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = key_of(t, i, j)
                if k not in keys:
                    lam = np.array([n - i - j, i, j]) / n
                    keys[k] = len(verts_P)
                    verts_P.append(lam[0] * V0[a] + lam[1] * V0[b] + lam[2] * V0[c])
                    h = lam[0] * PH[0] + lam[1] * PH[1] + lam[2] * PH[2]
                    verts_H.append((t, h / np.sqrt(h[2] ** 2 - h[0] ** 2 - h[1] ** 2)))
                idx[(i, j)] = keys[k]
        for (i, j), v in idx.items(): grid[(t, i, j)] = v
        for i in range(n):
            for j in range(n - i):
                faces.append((idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)])); face_tri.append(t)
                if i + j < n - 1:
                    faces.append((idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)])); face_tri.append(t)
    F = np.array(faces); XP = np.array(verts_P)
    # hyperbolic edge lengths: both endpoints expressed in the frame of a common original triangle
    pts_in = {}
    for t in range(56):
        a, b, c = orient[t]; PH = [lift(pos[t][v]) for v in (a, b, c)]
        for i in range(n + 1):
            for j in range(n + 1 - i):
                lam = np.array([n - i - j, i, j]) / n
                h = lam[0] * PH[0] + lam[1] * PH[1] + lam[2] * PH[2]
                pts_in[(t, keys[key_of(t, i, j)])] = h / np.sqrt(h[2] ** 2 - h[0] ** 2 - h[1] ** 2)
    E = {}
    for f, t in zip(F, face_tri):
        for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            e = (min(u, v), max(u, v))
            if e in E: continue
            p, q = pts_in[(t, u)], pts_in[(t, v)]
            E[e] = np.arccosh(max(1.0, p[2] * q[2] - p[0] * q[0] - p[1] * q[1]))
    edges = np.array(list(E.keys())); lenH = np.array(list(E.values()))
    # tau on fine vertices
    inv = {}
    for k, v in keys.items(): inv[v] = k
    def tau_key(k):
        if k[0] == 'v': return ('v', perm0[k[1]])
        if k[0] == 'e':
            _, p, q, lq = k; tp, tq = perm0[p], perm0[q]
            return ('e', tp, tq, lq) if tp < tq else ('e', tq, tp, n - lq)
        return ('f', frozenset((perm0[c], l) for c, l in k[1]))
    tau = np.array([keys[tau_key(inv[v])] for v in range(len(XP))])
    T = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1.]])
    assert np.abs(XP[tau] - XP @ T.T).max() < 1e-9, 'tau must act as T on P'
    return dict(F=F, XP=XP, edges=edges, lenH=lenH, tau=tau, level=level, face_tri=np.array(face_tri), grid=grid, n=n)

def prolongation(Mc, Mf):
    """sparse-free prolongation: returns (ia, ib) so that X_fine = (X_coarse[ia] + X_coarse[ib]) / 2"""
    nV = len(Mf['XP']); ia = np.full(nV, -1); ib = np.full(nV, -1)
    for (t, i, j), v in Mf['grid'].items():
        if ia[v] >= 0: continue
        g = Mc['grid']
        if i % 2 == 0 and j % 2 == 0: a = b = g[(t, i // 2, j // 2)]
        elif i % 2 == 1 and j % 2 == 0: a, b = g[(t, (i - 1) // 2, j // 2)], g[(t, (i + 1) // 2, j // 2)]
        elif i % 2 == 0 and j % 2 == 1: a, b = g[(t, i // 2, (j - 1) // 2)], g[(t, i // 2, (j + 1) // 2)]
        else: a, b = g[(t, (i + 1) // 2, (j - 1) // 2)], g[(t, (i - 1) // 2, (j + 1) // 2)]
        ia[v], ib[v] = a, b
    assert (ia >= 0).all()
    return ia, ib

if __name__ == '__main__':
    for l in range(0, 4):
        M = build(l); V, E, F = len(M['XP']), len(M['edges']), len(M['F'])
        print(f'level {l}: V={V} E={E} F={F} chi={V-E+F}  hyperbolic edge length {M["lenH"].min():.4f}..{M["lenH"].max():.4f}',
              ' tau^4=id:', np.all(M['tau'][M['tau'][M['tau'][M['tau']]]] == np.arange(V)))
