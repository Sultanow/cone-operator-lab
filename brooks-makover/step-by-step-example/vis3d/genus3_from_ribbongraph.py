# genus3_from_ribbongraph_final.py
#
# Worked example from the mini-paper:
#   n = 7, seed = 12
#   14 ideal triangles
#   21 side pairings
#   cusp cycles: 24, 14, 4
#   compactified genus = 3
#
# IMPORTANT:
# The combinatorics/topology are exactly those of the ribbon graph
# from the mini-paper. The 3D embedding is a visualization in R^3,
# not an isometric embedding of the hyperbolic metric.
#
# Install:
#   pip install numpy scipy pyvista
#
# Run:
#   python genus3_from_ribbongraph_final.py
#
# Outputs:
#   genus3_ribbon_surface_smooth.png
#   genus3_ribbon_surface_smooth.obj
#   genus3_ribbon_surface_smooth.stl

import numpy as np
import pyvista as pv

from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from collections import defaultdict, deque


# ============================================================
# 0. CONFIGURATION
# ============================================================

N_TRI = 14
N_DARTS = 3 * N_TRI

TRUNCATION_EPS = 0.22

# Core shaping
CORE_RELAX_ITERATIONS = 40
CORE_RELAX_SELF_WEIGHT = 0.72
CORE_RELAX_NEIGHBOR_WEIGHT = 0.28

# Initial cusp-mouth displacement
CUSP_PUSH = 1.15
CUSP_FLARE = 0.65

# Smooth trumpet extension
CUSP_NECK_LAYERS = 14
CUSP_TOTAL_LENGTH = 2.15

# Mesh refinement + smoothing
SUBDIVISION_LEVEL = 2
SMOOTH_ITERATIONS_1 = 35
SMOOTH_RELAXATION = 0.015
TAUBIN_ITERATIONS = 80
TAUBIN_PASS_BAND = 0.06

# Rendering
WINDOW_SIZE = (2200, 1600)

PNG_OUT = "genus3_ribbon_surface_smooth.png"
OBJ_OUT = "genus3_ribbon_surface_smooth.obj"
STL_OUT = "genus3_ribbon_surface_smooth.stl"


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
    """
    Cyclic order at each trivalent vertex:
        0 -> 1 -> 2 -> 0
    """
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

phi_cycle_lengths = sorted(
    [len(c) for c in phi_cycles],
    reverse=True
)

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

g_from_cusped = int(
    (2 - N_CUSPS - chi_cusped) / 2
)

print(
    f"F={F}, E={E}, cusp-count={N_CUSPS}, "
    f"chi_compact={chi_compact}, "
    f"chi_cusped={chi_cusped}, "
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

local_hex = np.array([
    v0, v1, v2, v3, v4, v5
])

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

    uf.union(
        local_vid(t1, x1),
        local_vid(t2, y2)
    )

    uf.union(
        local_vid(t1, y1),
        local_vid(t2, x2)
    )


rep_to_gid = {}


def global_id(triangle, local):

    rep = uf.find(
        local_vid(triangle, local)
    )

    if rep not in rep_to_gid:
        rep_to_gid[rep] = len(rep_to_gid)

    return rep_to_gid[rep]


# ============================================================
# 6. INITIAL 2D LAYOUT
# ============================================================

theta = np.linspace(
    0.0,
    2.0 * np.pi,
    N_TRI,
    endpoint=False
)


def rotmat(angle):

    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, -s],
        [s,  c]
    ])


triangle_centers_2d = []

for th in theta:

    radius = (
        4.0
        + 0.45 * np.sin(3.0 * th)
        + 0.15 * np.sin(5.0 * th)
    )

    triangle_centers_2d.append(
        np.array([
            radius * np.cos(th),
            radius * np.sin(th)
        ])
    )

triangle_centers_2d = np.asarray(
    triangle_centers_2d
)


local_positions_2d = {}

for t in range(N_TRI):

    ang = (
        theta[t]
        + 0.30 * np.sin(2.0 * theta[t])
        + 0.10 * np.cos(5.0 * theta[t])
    )

    R = rotmat(ang)

    for i in range(6):

        q = local_hex[i] - local_center

        p = (
            triangle_centers_2d[t]
            + 1.25 * (R @ q)
        )

        local_positions_2d[(t, i)] = p

    local_positions_2d[(t, 6)] = (
        triangle_centers_2d[t]
    )


# ============================================================
# 7. GLOBAL VERTEX POSITIONS
# ============================================================

gid_to_pts = defaultdict(list)

for t in range(N_TRI):

    for loc in range(N_LOCAL):

        gid = global_id(t, loc)

        gid_to_pts[gid].append(
            local_positions_2d[(t, loc)]
        )


n_global_vertices = len(gid_to_pts)

V2 = np.zeros(
    (n_global_vertices, 2),
    dtype=float
)

for gid, pts in gid_to_pts.items():

    V2[gid] = np.mean(
        np.asarray(pts),
        axis=0
    )


# ============================================================
# 8. TRIANGULATE HEXAGONS
# ============================================================

faces = []

for t in range(N_TRI):

    center = global_id(t, 6)

    boundary = [
        global_id(t, i)
        for i in range(6)
    ]

    for i in range(6):

        a = boundary[i]
        b = boundary[(i + 1) % 6]

        if len({center, a, b}) == 3:
            faces.append(
                (center, a, b)
            )


# ============================================================
# 9. EDGE INCIDENCE
# ============================================================

edge_count = defaultdict(int)

for a, b, c in faces:

    for u, v in [
        (a, b),
        (b, c),
        (c, a)
    ]:

        if u > v:
            u, v = v, u

        edge_count[(u, v)] += 1


boundary_edges = [
    edge
    for edge, count in edge_count.items()
    if count == 1
]


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


print(
    "boundary components:",
    len(boundary_components)
)

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


adjacency = coo_matrix(
    (data, (rows, cols)),
    shape=(
        n_global_vertices,
        n_global_vertices
    )
).tocsr()


degree = np.asarray(
    adjacency.sum(axis=1)
).ravel()


laplacian = (
    diags(degree)
    - adjacency
)


# ============================================================
# 12. SPECTRAL 3D EMBEDDING
# ============================================================

evals, evecs = eigsh(
    laplacian,
    k=4,
    which="SM"
)

order = np.argsort(evals)

evals = evals[order]
evecs = evecs[:, order]

X = evecs[:, 1:4].copy()

X -= X.mean(axis=0)

norm = np.max(
    np.linalg.norm(
        X,
        axis=1
    )
)

X /= norm


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

        avg = np.mean(
            X[nbrs],
            axis=0
        )

        Xnew[i] = (
            CORE_RELAX_SELF_WEIGHT * X[i]
            + CORE_RELAX_NEIGHBOR_WEIGHT * avg
        )

    X = Xnew


# ============================================================
# 15. PUSH + FLARE ORIGINAL CUSP MOUTHS
# ============================================================

def flare_boundary_component(
    X,
    component,
    push=CUSP_PUSH,
    flare=CUSP_FLARE
):

    comp = np.asarray(
        component,
        dtype=int
    )

    global_center = X.mean(axis=0)

    pts = X[comp]

    center = pts.mean(axis=0)

    direction = center - global_center

    nrm = np.linalg.norm(direction)

    if nrm < 1e-14:
        return X

    direction /= nrm

    new_center = (
        center
        + push * direction
    )

    rel = pts - center

    X[comp] = (
        new_center
        + (1.0 + flare) * rel
    )

    return X


for comp in boundary_components:

    X = flare_boundary_component(
        X,
        comp
    )


# ============================================================
# 16. SCALE CORE
# ============================================================

X[:, 0] *= 3.4
X[:, 1] *= 2.9
X[:, 2] *= 2.7


# ============================================================
# 17. ORDER BOUNDARY LOOPS
# ============================================================

def order_boundary_loop(
    component,
    bd_adj
):

    component = set(component)

    start = next(iter(component))

    loop = [start]

    previous = None
    current = start

    while True:

        candidates = [
            x
            for x in bd_adj[current]
            if x in component
            and x != previous
        ]

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
    return (
        t * t
        * (3.0 - 2.0 * t)
    )


base_vertices = X.tolist()

extended_faces = list(faces)


global_center = X.mean(axis=0)


for comp_index, comp in enumerate(boundary_components):

    loop = order_boundary_loop(
        comp,
        bd_adj
    )

    loop = np.asarray(
        loop,
        dtype=int
    )

    base_ring = X[loop]

    ring_center = (
        base_ring.mean(axis=0)
    )

    direction = (
        ring_center
        - global_center
    )

    direction /= np.linalg.norm(
        direction
    )

    rel = (
        base_ring
        - ring_center
    )

    previous_ring = loop.tolist()

    for layer in range(
        1,
        CUSP_NECK_LAYERS + 1
    ):

        t = (
            layer
            / CUSP_NECK_LAYERS
        )

        # Smooth longitudinal growth.
        axial_profile = (
            0.30 * t
            + 0.70 * t**1.85
        )

        axial = (
            CUSP_TOTAL_LENGTH
            * axial_profile
        )

        displacement = (
            axial
            * direction
        )

        # Smooth trumpet flare.
        s = smoothstep(t)

        scale = (
            1.0
            + 0.10 * t
            + 1.35 * s**1.45
        )

        # Slight asymmetric variation:
        # keeps it from looking like a perfect artificial cone.
        asym = (
            1.0
            + 0.035
            * np.sin(
                2.0 * np.pi * t
                + comp_index
            )
        )

        new_ring_pts = (
            ring_center
            + displacement
            + scale * asym * rel
        )

        new_ring = []

        for p in new_ring_pts:

            idx = len(base_vertices)

            base_vertices.append(
                p.tolist()
            )

            new_ring.append(idx)

        m = len(previous_ring)

        for j in range(m):

            a = previous_ring[j]

            b = previous_ring[
                (j + 1) % m
            ]

            c = new_ring[
                (j + 1) % m
            ]

            d = new_ring[j]

            extended_faces.append(
                (a, b, c)
            )

            extended_faces.append(
                (a, c, d)
            )

        previous_ring = new_ring


X_extended = np.asarray(
    base_vertices,
    dtype=float
)


# ============================================================
# 19. PYVISTA FACE ARRAY
# ============================================================

face_array = []

for a, b, c in extended_faces:

    face_array.extend([
        3,
        int(a),
        int(b),
        int(c)
    ])


mesh = pv.PolyData(
    X_extended,
    face_array
)


# ============================================================
# 20. HIGH-QUALITY SURFACE REFINEMENT
# ============================================================

mesh = mesh.clean()

mesh = mesh.triangulate()


print(
    "Original mesh:",
    mesh.n_points,
    "points,",
    mesh.n_cells,
    "triangles"
)


# Loop subdivision creates real geometric resolution.
mesh = mesh.subdivide(
    SUBDIVISION_LEVEL,
    subfilter="loop"
)


print(
    "After subdivision:",
    mesh.n_points,
    "points,",
    mesh.n_cells,
    "triangles"
)


# First smoothing:
# remove local angularity.
mesh = mesh.smooth(
    n_iter=SMOOTH_ITERATIONS_1,
    relaxation_factor=SMOOTH_RELAXATION,
    feature_smoothing=False,
    boundary_smoothing=True
)


# Second smoothing:
# preserve overall shape while removing residual faceting.
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
# 22. PAPER-STYLE INTERACTIVE RENDER
# ============================================================

plotter = pv.Plotter(
    window_size=WINDOW_SIZE,
    off_screen=False
)

plotter.set_background(
    "white"
)


plotter.add_mesh(
    mesh,
    color="#f5f5f1",
    smooth_shading=True,

    # Fine mathematical triangulation.
    show_edges=True,
    edge_color="#737373",
    line_width=0.45,

    ambient=0.32,
    diffuse=0.68,

    specular=0.12,
    specular_power=18
)


plotter.enable_lightkit()


# ============================================================
# 23. CAMERA
# ============================================================

plotter.camera_position = [
    (10.5, -8.5, 6.4),
    (0.0, 0.0, 0.15),
    (0.0, 0.0, 1.0)
]

plotter.camera.zoom(1.05)


# ============================================================
# 24. INTERACTION
# ============================================================

print()
print("Interactive controls:")
print("  left mouse   : rotate")
print("  middle mouse : pan")
print("  mouse wheel  : zoom")
print("  q            : close")
print()
print(f"Screenshot: {PNG_OUT}")
print()


# ============================================================
# 25. DISPLAY + SCREENSHOT
# ============================================================

plotter.show(
    screenshot=PNG_OUT,
    auto_close=False
)

print()
print(f"Saved: {PNG_OUT}")