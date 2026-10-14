"""Length spectra of triangulated hyperbolic surfaces given only by edge lengths.
Develop the triangles into the hyperboloid model along a spanning tree of the dual graph;
every non-tree edge yields a Lorentz transformation (side pairing). Closed loops of
coarse triangles are realised as fine dual paths; their holonomy M in SO+(2,1) gives the
length of the closed geodesic in their free homotopy class:  trace M = 1 + 2 cosh(length)."""
import numpy as np
from collections import deque
J = np.diag([1.0, 1.0, -1.0])

def angle(a, b, c):  # hyperbolic angle opposite side a
    return np.arccos(np.clip((np.cosh(b) * np.cosh(c) - np.cosh(a)) / (np.sinh(b) * np.sinh(c)), -1, 1))

def place(pv, pu, L_vw, alpha):
    """point at distance L_vw from pv, at angle alpha (counter-clockwise) from the direction pv->pu"""
    e0 = pv; t = pu + (pu @ J @ pv) * pv; e1 = t / np.sqrt(t @ J @ t); e2 = J @ np.cross(e0, e1)
    return np.cosh(L_vw) * e0 + np.sinh(L_vw) * (np.cos(alpha) * e1 + np.sin(alpha) * e2)

class Developed:
    def __init__(self, F, length):  # F oriented faces, length(i,j) -> hyperbolic length
        self.F = F
        self.edge_faces = {}
        for f, (a, b, c) in enumerate(F):
            for u, v in ((a, b), (b, c), (c, a)): self.edge_faces.setdefault(frozenset((u, v)), []).append(f)
        pos = {}; a, b, c = F[0]
        lab, lac, lbc = length(a, b), length(a, c), length(b, c)
        pos[0] = {a: np.array([0.0, 0.0, 1.0]), b: np.array([np.sinh(lab), 0, np.cosh(lab)])}
        pos[0][c] = place(pos[0][a], pos[0][b], lac, angle(lbc, lab, lac))
        self.tree = set(); q = deque([0])
        def copy_across(f, u, v, g):
            w = [x for x in F[g] if x not in (u, v)][0]
            return {v: pos[f][v], u: pos[f][u], w: place(pos[f][v], pos[f][u], length(v, w), angle(length(u, w), length(u, v), length(v, w)))}
        self.copy_across = copy_across
        while q:
            f = q.popleft(); a, b, c = F[f]
            for u, v in ((a, b), (b, c), (c, a)):
                for g in self.edge_faces[frozenset((u, v))]:
                    if g == f or g in pos: continue
                    pos[g] = copy_across(f, u, v, g); self.tree.add(frozenset((f, g))); q.append(g)
        self.pos = pos; self.cache = {}
    def transition(self, f, g):
        if frozenset((f, g)) in self.tree: return np.eye(3)
        if (f, g) not in self.cache:
            shared = set(self.F[f]) & set(self.F[g]); a, b, c = self.F[f]
            u, v = [(x, y) for x, y in ((a, b), (b, c), (c, a)) if {x, y} == shared][0]
            cp = self.copy_across(f, u, v, g); vs = list(self.F[g])
            Q = np.stack([self.pos[g][x] for x in vs], 1); C = np.stack([cp[x] for x in vs], 1)
            self.cache[(f, g)] = Q @ np.linalg.inv(C)
        return self.cache[(f, g)]
    def holonomy(self, faces):
        M = np.eye(3)
        for f, g in zip(faces, faces[1:] + faces[:1]):
            if f != g: M = M @ np.linalg.inv(self.transition(f, g))
        return M

def length_of(M):
    return float(np.arccosh(max(1.0, (np.trace(M) - 1) / 2)))
