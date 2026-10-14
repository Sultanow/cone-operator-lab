# genus3_from_ribbongraph_refined.py
#
# Refined 3D visualization of the exact genus-3 ribbon surface
# from the worked mini-paper (n=7, seed=12).
#
# Install:
#   pip install numpy scipy pyvista
#
# Run:
#   python genus3_from_ribbongraph_refined.py
#
# Outputs:
#   genus3_ribbon_surface_refined.png
#   genus3_ribbon_surface_refined.obj
#   genus3_ribbon_surface_refined.stl

import numpy as np
import pyvista as pv

from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
from collections import defaultdict, deque


# ============================================================
# 0. CONFIGURATION
# ============================================================

N_TRI = 14
N_DARTS = 3 * N_TRI

TRUNCATION_EPS = 0.22

# Core smoothing / relaxation
CORE_RELAX_ITERATIONS = 60
CORE_RELAX_SELF_WEIGHT = 0.72
CORE_RELAX_NEIGHBOR_WEIGHT = 0.28

# Extra geometric refinement of embedding
REPULSION_STEPS = 300
REPULSION_RADIUS = 0.78
REPULSION_STRENGTH = 0.018
SPRING_STRENGTH = 0.055

CUSP_PUSH = 1.40
CUSP_FLARE = 0.82

CUSP_NECK_LAYERS = 40
CUSP_TOTAL_LENGTH = 2.75

# level 4 = 4^4 = 256 times as many triangles as base mesh
SUBDIVISION_LEVEL = 4

SMOOTH_ITERATIONS_1 = 60
SMOOTH_RELAXATION = 0.010

TAUBIN_ITERATIONS = 160
TAUBIN_PASS_BAND = 0.045

# high-res paper rendering
WINDOW_SIZE = (3200, 2400)

PNG_OUT = "genus3_ribbon_surface_refined.png"
OBJ_OUT = "genus3_ribbon_surface_refined.obj"
STL_OUT = "genus3_ribbon_surface_refined.stl"


# ============================================================
# 1. EXACT RIBBON-GRAPH DATA
# ============================================================

TAU_PAIRS = [
    (0, 29),
    (1, 6),
    (2, 4),
    (3, 17),
    (5, 21),
    (7, 15),
    (8, 23),
    (9, 14),
    (10, 27),
    (11, 33),
    (12, 41),
    (13, 39),
    (16, 35),
    (18, 32),
    (19, 30),
    (20, 37),
    (22, 36),
    (24, 31),
    (25, 38),
    (26, 34),
    (28, 40),
]


# ============================================================
# 2. RIBBON PERMUTATIONS
# ============================================================

def sigma(d):
    v, s = divmod(d, 3)
    return 3 * v + ((s + 1) % 3)


tau = [-1] * N_DARTS
for a, b in TAU_PAIRS:
    tau[a] = b
    tau[b] = a

assert all(t >= 0 for t in tau), "Incomplete tau pairing."

phi = [tau[sigma(d)] for d in range(N_DARTS)]


def permutation_cycles(p):
    seen = [False] * len(p)
    cycles = []
    for i in range(len(p)):
        if seen[i]:
            continue
        cyc = []
        j = i
        while not seen[j]:
            seen[j] = True
            cyc.append(j)
            j = p[j]
        cycles.append(cyc)
    return cycles


phi_cycles = permutation_cycles(phi)
phi_cycle_lengths = sorted([len(c) for c in phi_cycles], reverse=True)

print("phi-cycle lengths:", phi_cycle_lengths)
assert phi_cycle_lengths == [24, 14, 4]


# ============================================================
# 3. TOPOLOGY CHECK
# ============================================================

F = N_TRI
E = len(TAU_PAIRS)
N_CUSPS = len(phi_cycles)

chi_compact = N_CUSPS - E + F
chi_cusped = F - E

g_from_compact = int((2 - chi_compact) / 2)
g_from_cusped = int((2 - N_CUSPS - chi_cusped) / 2)

print(
    f"F={F}, E={E}, cusp-count={N_CUSPS}, "
    f"chi_compact={chi_compact}, chi_cusped={chi_cusped}, "
    f"genus={g_from_compact}"
)

assert chi_compact == -4
assert chi_cusped == -7
assert g_from_compact == 3
assert g_from_cusped == 3


# ============================================================
# 4. LOCAL TRUNCATED TRIANGLE
# ============================================================

sqrt3 = np.sqrt(3.0)

A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, sqrt3 / 2.0])

eps = TRUNCATION_EPS

v0 = (1 - eps) * A + eps * B
v1 = eps * A + (1 - eps) * B

v2 = (1 - eps) * B + eps * C
v3 = eps * B + (1 - eps) * C

v4 = (1 - eps) * C + eps * A
v5 = eps * C + (1 - eps) * A

local_hex = np.array([v0, v1, v2, v3, v4, v5])
local_center = local_hex.mean(axis=0)

SIDE_TO_LOCAL_EDGE = {
    0: (0, 1),
    1: (2, 3),
    2: (4, 5),
}


# ============================================================
# 5. UNION-FIND
# ============================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


N_LOCAL = 7


def local_vid(triangle, local):
    return N_LOCAL * triangle + local


uf = UnionFind(N_LOCAL * N_TRI)

for a, b in TAU_PAIRS:
    t1, s1 = divmod(a, 3)
    t2, s2 = divmod(b, 3)

    x1, y1 = SIDE_TO_LOCAL_EDGE[s1]
    x2, y2 = SIDE_TO_LOCAL_EDGE[s2]

    uf.union(local_vid(t1, x1), local_vid(t2, y2))
    uf.union(local_vid(t1, y1), local_vid(t2, x2))

rep_to_gid = {}


def global_id(triangle, local):
    rep = uf.find(local_vid(triangle, local))
    if rep not in rep_to_gid:
        rep_to_gid[rep] = len(rep_to_gid)
    return rep_to_gid[rep]


# ============================================================
# 6. INITIAL 2D LAYOUT
# ============================================================

theta = np.linspace(0.0, 2.0 * np.pi, N_TRI, endpoint=False)


def rotmat(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]])


triangle_centers_2d = []
for th in theta:
    radius = 4.0 + 0.45 * np.sin(3.0 * th) + 0.15 * np.sin(5.0 * th)
    triangle_centers_2d.append(np.array([radius * np.cos(th), radius * np.sin(th)]))
triangle_centers_2d = np.asarray(triangle_centers_2d)

local_positions_2d = {}
for t in range(N_TRI):
    ang = theta[t] + 0.30 * np.sin(2.0 * theta[t]) + 0.10 * np.cos(5.0 * theta[t])
    R = rotmat(ang)

    for i in range(6):
        q = local_hex[i] - local_center
        p = triangle_centers_2d[t] + 1.25 * (R @ q)
        local_positions_2d[(t, i)] = p

    local_positions_2d[(t, 6)] = triangle_centers_2d[t]


# ============================================================
# 7. GLOBAL VERTEX POSITIONS
# ============================================================

gid_to_pts = defaultdict(list)

for t in range(N_TRI):
    for loc in range(N_LOCAL):
        gid = global_id(t, loc)
        gid_to_pts[gid].append(local_positions_2d[(t, loc)])

n_global_vertices = len(gid_to_pts)

V2 = np.zeros((n_global_vertices, 2), dtype=float)
for gid, pts in gid_to_pts.items():
    V2[gid] = np.mean(np.asarray(pts), axis=0)


# ============================================================
# 8. TRIANGULATE HEXAGONS
# ============================================================

faces = []

for t in range(N_TRI):
    center = global_id(t, 6)
    boundary = [global_id(t, i) for i in range(6)]

    for i in range(6):
        a = boundary[i]
        b = boundary[(i + 1) % 6]
        if len({center, a, b}) == 3:
            faces.append((center, a, b))


# ============================================================
# 9. EDGE INCIDENCE
# ============================================================

edge_count = defaultdict(int)

for a, b, c in faces:
    for u, v in [(a, b), (b, c), (c, a)]:
        if u > v:
            u, v = v, u
        edge_count[(u, v)] += 1

boundary_edges = [edge for edge, count in edge_count.items() if count == 1]


# ============================================================
# 10. BOUNDARY COMPONENTS
# ============================================================

bd_adj = defaultdict(list)
for u, v in boundary_edges:
    bd_adj[u].append(v)
    bd_adj[v].append(u)

boundary_components = []
seen = set()

for start in bd_adj:
    if start in seen:
        continue
    queue = deque([start])
    seen.add(start)
    comp = []

    while queue:
        x = queue.popleft()
        comp.append(x)
        for y in bd_adj[x]:
            if y not in seen:
                seen.add(y)
                queue.append(y)

    boundary_components.append(comp)

print("boundary components:", len(boundary_components))
assert len(boundary_components) == 3


# ============================================================
# 11. GRAPH LAPLACIAN
# ============================================================

rows = []
cols = []
data = []

for u, v in edge_count:
    rows += [u, v]
    cols += [v, u]
    data += [1.0, 1.0]

adjacency = coo_matrix((data, (rows, cols)), shape=(n_global_vertices, n_global_vertices)).tocsr()
degree = np.asarray(adjacency.sum(axis=1)).ravel()
laplacian = diags(degree) - adjacency


# ============================================================
# 12. SPECTRAL 3D EMBEDDING
# ============================================================

evals, evecs = eigsh(laplacian, k=4, which="SM")
order = np.argsort(evals)
evals = evals[order]
evecs = evecs[:, order]

X = evecs[:, 1:4].copy()
X -= X.mean(axis=0)
X /= np.max(np.linalg.norm(X, axis=1))


# ============================================================
# 13. VERTEX ADJACENCY
# ============================================================

neighbors = defaultdict(list)
for u, v in edge_count:
    neighbors[u].append(v)
    neighbors[v].append(u)

boundary_vertex_set = set()
for comp in boundary_components:
    boundary_vertex_set.update(comp)


# ============================================================
# 14. RELAX CORE
# ============================================================

for _ in range(CORE_RELAX_ITERATIONS):
    Xnew = X.copy()
    for i in range(n_global_vertices):
        if i in boundary_vertex_set:
            continue
        nbrs = neighbors[i]
        if not nbrs:
            continue
        avg = np.mean(X[nbrs], axis=0)
        Xnew[i] = CORE_RELAX_SELF_WEIGHT * X[i] + CORE_RELAX_NEIGHBOR_WEIGHT * avg
    X = Xnew


# ============================================================
# 15. SHORT-RANGE REPULSION TO REDUCE INTERSECTIONS
# ============================================================

edge_lengths = []
for u, v in edge_count:
    edge_lengths.append(np.linalg.norm(X[u] - X[v]))
target_edge = float(np.median(edge_lengths))

adjacent_pairs = set()
for u, v in edge_count:
    adjacent_pairs.add((min(u, v), max(u, v)))

for _ in range(REPULSION_STEPS):
    force = np.zeros_like(X)

    # spring term on edges
    for u, v in edge_count:
        d = X[v] - X[u]
        dist = np.linalg.norm(d) + 1e-12
        delta = dist - target_edge
        f = SPRING_STRENGTH * delta * d / dist
        force[u] += f
        force[v] -= f

    # repulsion for nearby non-neighbors
    tree = cKDTree(X)
    pairs = tree.query_pairs(REPULSION_RADIUS)

    for i, j in pairs:
        if (min(i, j), max(i, j)) in adjacent_pairs:
            continue
        d = X[j] - X[i]
        dist = np.linalg.norm(d) + 1e-12
        if dist < REPULSION_RADIUS:
            mag = REPULSION_STRENGTH * (REPULSION_RADIUS - dist) / dist
            f = -mag * d
            force[i] += f
            force[j] -= f

    # freeze boundary loops only partially, move interior more
    X += force * 0.22

    # recenter and renormalize softly
    X -= X.mean(axis=0)
    X /= max(np.max(np.linalg.norm(X, axis=1)), 1e-12)


# ============================================================
# 16. PUSH + FLARE ORIGINAL CUSP MOUTHS
# ============================================================

def flare_boundary_component(X, component, push=CUSP_PUSH, flare=CUSP_FLARE):
    comp = np.asarray(component, dtype=int)
    global_center = X.mean(axis=0)

    pts = X[comp]
    center = pts.mean(axis=0)

    direction = center - global_center
    nrm = np.linalg.norm(direction)
    if nrm < 1e-14:
        return X
    direction /= nrm

    new_center = center + push * direction
    rel = pts - center

    X[comp] = new_center + (1.0 + flare) * rel
    return X


for comp in boundary_components:
    X = flare_boundary_component(X, comp)


# scale core
X[:, 0] *= 3.6
X[:, 1] *= 3.1
X[:, 2] *= 2.95


# ============================================================
# 17. ORDER BOUNDARY LOOPS
# ============================================================

def order_boundary_loop(component, bd_adj):
    component = set(component)
    start = next(iter(component))

    loop = [start]
    previous = None
    current = start

    while True:
        candidates = [x for x in bd_adj[current] if x in component and x != previous]
        if not candidates:
            break

        nxt = candidates[0]
        if nxt == start:
            break

        loop.append(nxt)
        previous = current
        current = nxt

        if len(loop) > len(component) + 2:
            break

    return loop


# ============================================================
# 18. SMOOTH TRUMPET PROFILE
# ============================================================

def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


base_vertices = X.tolist()
extended_faces = list(faces)

global_center = X.mean(axis=0)

for comp_index, comp in enumerate(boundary_components):
    loop = order_boundary_loop(comp, bd_adj)
    loop = np.asarray(loop, dtype=int)

    base_ring = X[loop]
    ring_center = base_ring.mean(axis=0)

    direction = ring_center - global_center
    direction /= np.linalg.norm(direction)

    rel = base_ring - ring_center
    previous_ring = loop.tolist()

    for layer in range(1, CUSP_NECK_LAYERS + 1):
        t = layer / CUSP_NECK_LAYERS

        axial_profile = 0.25 * t + 0.75 * t**1.9
        axial = CUSP_TOTAL_LENGTH * axial_profile
        displacement = axial * direction

        s = smoothstep(t)
        scale = 1.0 + 0.08 * t + 1.45 * s**1.55

        asym = 1.0 + 0.025 * np.sin(2.0 * np.pi * t + 0.7 * comp_index)

        new_ring_pts = ring_center + displacement + scale * asym * rel

        new_ring = []
        for p in new_ring_pts:
            idx = len(base_vertices)
            base_vertices.append(p.tolist())
            new_ring.append(idx)

        m = len(previous_ring)
        for j in range(m):
            a = previous_ring[j]
            b = previous_ring[(j + 1) % m]
            c = new_ring[(j + 1) % m]
            d = new_ring[j]

            extended_faces.append((a, b, c))
            extended_faces.append((a, c, d))

        previous_ring = new_ring


X_extended = np.asarray(base_vertices, dtype=float)


# ============================================================
# 19. PYVISTA FACE ARRAY
# ============================================================

face_array = []
for a, b, c in extended_faces:
    face_array.extend([3, int(a), int(b), int(c)])

mesh = pv.PolyData(X_extended, face_array)


# ============================================================
# 20. HIGH-QUALITY SURFACE REFINEMENT
# ============================================================

mesh = mesh.clean()
mesh = mesh.triangulate()

print("Original mesh:", mesh.n_points, "points,", mesh.n_cells, "triangles")

mesh = mesh.subdivide(SUBDIVISION_LEVEL, subfilter="loop")

print("After subdivision:", mesh.n_points, "points,", mesh.n_cells, "triangles")

mesh = mesh.smooth(
    n_iter=SMOOTH_ITERATIONS_1,
    relaxation_factor=SMOOTH_RELAXATION,
    feature_smoothing=False,
    boundary_smoothing=True
)

mesh = mesh.smooth_taubin(
    n_iter=TAUBIN_ITERATIONS,
    pass_band=TAUBIN_PASS_BAND
)

mesh = mesh.compute_normals(
    auto_orient_normals=True,
    split_vertices=False,
    inplace=False
)


# ============================================================
# 21. EXPORT
# ============================================================

mesh.save(OBJ_OUT)
mesh.save(STL_OUT)

print()
print(f"Saved: {OBJ_OUT}")
print(f"Saved: {STL_OUT}")


# ============================================================
# 22. LUXURY PAPER RENDERER
# ============================================================

def configure_plotter(plotter, mesh):
    """
    Common publication-quality rendering setup.
    """

    plotter.set_background("white")

    plotter.add_mesh(
        mesh,

        # warm mathematical white
        color="#f5f5f0",

        smooth_shading=True,

        # show triangulation, but very subtly
        show_edges=True,
        edge_color="#858585",
        line_width=0.25,

        ambient=0.30,
        diffuse=0.72,

        specular=0.18,
        specular_power=28,
    )

    plotter.enable_lightkit()

    # High-quality anti-aliasing
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass

    # Correct clipping for deep tunnels
    plotter.camera.clipping_range = (0.001, 1000.0)


# ============================================================
# 23. CAMERA PRESETS
# ============================================================

camera_views = {

    # Main paper image
    "hero": [
        (11.5, -9.5, 7.5),
        (0.0, 0.0, 0.1),
        (0.0, 0.0, 1.0),
    ],

    # More frontal view - useful for topology
    "front": [
        (0.0, -14.0, 3.8),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ],

    # Topological overview
    "top": [
        (0.0, 0.0, 16.0),
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ],

    # Opposite side - reveals hidden tunnels
    "reverse": [
        (-10.5, 9.0, 6.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ],
}


# ============================================================
# 24. AUTOMATIC HIGH-RES RENDERS
# ============================================================

print()
print("Generating luxury paper renders ...")
print()

for name, camera in camera_views.items():

    output = f"genus3_ribbon_surface_{name}.png"

    p = pv.Plotter(
        off_screen=True,
        window_size=WINDOW_SIZE
    )

    configure_plotter(
        p,
        mesh
    )

    p.camera_position = camera

    p.camera.zoom(1.10)

    # Force rendering once before screenshot
    p.render()

    p.screenshot(
        output,
        transparent_background=False
    )

    p.close()

    print(
        f"Saved: {output}"
    )


# ============================================================
# 25. CLEAN SURFACE VERSION
#
# Same hero perspective, but WITHOUT triangulation.
# Very useful as a second paper figure.
# ============================================================

clean_output = "genus3_ribbon_surface_hero_clean.png"

p = pv.Plotter(
    off_screen=True,
    window_size=WINDOW_SIZE
)

p.set_background("white")

p.add_mesh(
    mesh,
    color="#f5f5f0",
    smooth_shading=True,

    show_edges=False,

    ambient=0.30,
    diffuse=0.72,

    specular=0.20,
    specular_power=30,
)

p.enable_lightkit()

try:
    p.enable_anti_aliasing("ssaa")
except Exception:
    pass

p.camera_position = camera_views["hero"]

p.camera.zoom(1.10)

p.camera.clipping_range = (
    0.001,
    1000.0
)

p.render()

p.screenshot(
    clean_output,
    transparent_background=False
)

p.close()

print(
    f"Saved: {clean_output}"
)


# ============================================================
# 26. INTERACTIVE VIEWER
# ============================================================

print()
print("Opening interactive viewer ...")
print()
print("Controls:")
print("  left mouse   : rotate")
print("  middle mouse : pan")
print("  mouse wheel  : zoom")
print("  r            : reset camera")
print("  q            : close")
print()


plotter = pv.Plotter(
    window_size=(2200, 1600),
    off_screen=False
)

configure_plotter(
    plotter,
    mesh
)

plotter.camera_position = (
    camera_views["hero"]
)

plotter.camera.zoom(1.10)

plotter.show()


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print()
print("==============================================")
print(" Render complete")
print("==============================================")

print()
print("Generated:")
print("  genus3_ribbon_surface_hero.png")
print("  genus3_ribbon_surface_front.png")
print("  genus3_ribbon_surface_top.png")
print("  genus3_ribbon_surface_reverse.png")
print("  genus3_ribbon_surface_hero_clean.png")
print()

print(
    f"Mesh: {mesh.n_points:,} points, "
    f"{mesh.n_cells:,} triangles"
)