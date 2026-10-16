"""Interactive 3D viewer on the GPU (OpenGL, Polyscope) for Mizhaev's polyhedron and the surfaces of the sweeps.
Rotate with the left mouse button, pan with the right, zoom with the wheel; every object can be switched on and off
in the panel on the left. The panel 'Mizhaev' has buttons for screenshots (PNG, transparent background), for a
turntable sequence of frames, and for going back to the view of Fig. 7. The defaults follow Fig. 7 of the paper:
nonagon colours, white background, no edge lines, the 24 cone points on every surface coloured by their cone angle,
the same viewing direction in parallel projection, and with --side-by-side the surfaces in a row as in the figure.
With --material wax (or clay, candy, ...) Polyscope's own lighting replaces the colours of the paper.
    pip install polyscope
    python visual/view3d.py                               # P, its edges, the 24 cone points, the symmetry axis
    python visual/view3d.py visual/*_L4.json              # add surfaces exported with scripts/export_obj.py
    python visual/view3d.py A.json B.json --side-by-side  # place the surfaces next to P instead of on top of it
    python visual/view3d.py --turntable 72 --out frames/  # no window: 72 frames of a full turn, then exit"""
import os, sys, json, argparse, numpy as np
import polyscope as ps, polyscope.imgui as psim
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(os.path.dirname(HERE), 'src')
ap = argparse.ArgumentParser(); ap.add_argument('surfaces', nargs='*', help='JSON files from scripts/export_obj.py')
ap.add_argument('--side-by-side', action='store_true'); ap.add_argument('--ssaa', type=int, default=4, help='anti-aliasing factor')
ap.add_argument('--turntable', type=int, default=0, help='write this many frames of a full turn and exit')
ap.add_argument('--out', default='.', help='folder for screenshots and frames'); ap.add_argument('--backend', default=None)
ap.add_argument('--material', default=None, help='polyscope material (wax, clay, candy, ...): its own lighting instead of the colours of the paper')
ap.add_argument('--edges', action='store_true', help='show the edges of P'); ap.add_argument('--ground', action='store_true', help='shadow on the ground')
ap.add_argument('--perspective', action='store_true', help='perspective instead of the parallel projection of Fig. 7')
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
PAL = np.array([[0x4E, 0x79, 0xA7], [0xF2, 0x8E, 0x2B], [0xE1, 0x57, 0x59], [0x76, 0xB7, 0xB2],
                [0x59, 0xA1, 0x4F], [0xED, 0xC9, 0x48], [0xB0, 0x7A, 0xA1], [0xFF, 0x9D, 0xA7]]) / 255   # nonagon colours of the paper
sys.path.insert(0, SRC)
R = json.load(open(os.path.join(SRC, 'results.json'))); V = np.array(R['vertices'], float)
TRI = np.array(R['triangles']) - 1; FACES = [[x - 1 for x in f] for f in R['faces']]
tri_face = np.array([next(k for k, f in enumerate(FACES) if set(t) <= set(f)) for t in TRI])
cone = np.degrees(2 * np.pi - np.array(R['original_curvature']))             # cone angle at each vertex of P

# Colours as in the paper. Polyscope tone-maps every rendered colour (exposure 1, Reinhard curve with white level 0.75,
# gamma 2.2), which makes saturated colours look pale. Unless a material is chosen, faces are drawn with the 'flat'
# material, shaded per face like the figures (0.62 + 0.38 |n . light| with the light of Fig. 7, fixed in space), and
# every colour is passed through the inverse tone curve, so that the screen shows exactly the colours of the paper.
PAPER = args.material is None; MATERIAL = 'flat' if PAPER else args.material
LIGHT = np.array([0.1618, -0.3484, 0.9233])
def untonemap(c, white=0.75, gamma=2.2):
    y = np.clip(np.asarray(c, float), 0, 1) ** gamma; w2 = white ** 2
    return (-(1 - y) + np.sqrt((1 - y) ** 2 + 4 * y / w2)) / (2 / w2)        # solves x (1 + x / w^2) / (1 + x) = y
def face_colours(X, F, base):
    if not PAPER: return base
    n = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    return untonemap(base * (0.62 + 0.38 * np.abs(n @ LIGHT))[:, None])
try:
    from matplotlib import colormaps; _cw = colormaps['coolwarm']
    CONE_RGB = np.array([_cw(a / 720)[:3] for a in cone])
except Exception:
    CONE_RGB = None
edges = np.array(sorted({tuple(sorted((f[i], f[(i + 1) % 9]))) for f in FACES for i in range(9)}))

ps.init(args.backend) if args.backend else ps.init()
ps.set_program_name('Mizhaev polyhedron'); ps.set_up_dir('z_up'); ps.set_ground_plane_mode('shadow_only' if args.ground else 'none')
ps.set_background_color((1.0, 1.0, 1.0)); ps.set_SSAA_factor(args.ssaa); ps.set_transparency_mode('pretty')
if not args.perspective: ps.set_view_projection_mode('orthographic')
# the camera of Fig. 7 (elevation 22 deg, azimuth 40 deg in visual/fig_views.py): direction from the scene to the camera
# and the direction that appears as 'right' on the screen, along which --side-by-side places the surfaces
CAM = np.array([0.5960, 0.7103, -0.3746]); RIGHT = np.array([-0.7660, 0.6428, 0.0])
m = ps.register_surface_mesh('P (56 triangles of K)', V, TRI, smooth_shade=False, edge_width=0.0, material=MATERIAL)
m.add_color_quantity('nonagons', face_colours(V, TRI, PAL[tri_face]), defined_on='faces', enabled=True)
ps.register_curve_network('edges of P', V, edges, radius=0.0025, color=(0.11, 0.15, 0.2), enabled=args.edges)
def cone_points(name, X):
    pc = ps.register_point_cloud(name, X, radius=0.0075, material='flat')
    if PAPER and CONE_RGB is not None: pc.add_color_quantity('cone angle (coolwarm, 0-720 degrees)', untonemap(CONE_RGB), enabled=True)
    else: pc.add_scalar_quantity('cone angle (degrees)', cone, cmap='coolwarm', vminmax=(0, 720), enabled=True)
cone_points('cone points of P', V)
ps.register_curve_network('symmetry axis', np.array([[0, 0, -330.], [0, 0, 330.]]), np.array([[0, 1]]),
                          radius=0.0012, color=(0.5, 0.5, 0.5), enabled=False)
centres = [V.mean(0)]
for k, f in enumerate(args.surfaces):                                       # surfaces from the sweeps
    d = json.load(open(f)); X = np.array(d['vertices'], float); F = np.array(d['faces']) - 1; G = np.array(d['nonagon']) - 1
    if args.side_by_side: X = X + 640.0 * (k + 1) * RIGHT
    name = os.path.splitext(os.path.basename(f))[0]; on = args.side_by_side or k == 0
    s = ps.register_surface_mesh(name, X, F, smooth_shade=not PAPER, edge_width=0.0, material=MATERIAL, enabled=on)
    s.add_color_quantity('nonagons', face_colours(X, F, PAL[G]), defined_on='faces', enabled=True)
    try:                                                                    # the 24 cone points of P on this surface
        from fine_mesh import build
        lev = {24: 0, 108: 1, 444: 2, 1788: 3, 7164: 4}[len(X)]; XP = build(lev, SRC)['XP']
        marks = [int(np.argmin(np.linalg.norm(XP - V[i], axis=1))) for i in range(24)]
        cone_points(f'cone points on {name}', X[marks])
    except Exception as err:
        print(f'cone points on {name} skipped: {err}')
    if not args.side_by_side: m.set_transparency(0.35)                       # P shows through the overlaid surface
    else: centres.append(X.mean(0))
TARGET = np.mean(centres, axis=0); DIST = 1400.0 + 640.0 * (len(centres) - 1)
def fig7_view(): ps.look_at(tuple(TARGET + DIST * CAM), tuple(TARGET))
counter = [0]
def screenshot(name=None):
    counter[0] += 1; fn = os.path.join(args.out, name or f'mizhaev_{counter[0]:03d}.png')
    ps.screenshot(fn, transparent_bg=True); return fn
def turntable(n, radius=1100.0, height=520.0):
    for i in range(n):
        a = 2 * np.pi * i / n; ps.look_at((radius * np.cos(a), radius * np.sin(a), height), (0.0, 0.0, 0.0))
        screenshot(f'turntable_{i:03d}.png')
last = ['']
def panel():
    psim.PushItemWidth(150)
    psim.TextUnformatted('Mizhaev\'s polyhedron: 24 vertices, 36 edges, 8 nonagons, genus 3')
    if psim.Button('Screenshot (PNG, transparent)'): last[0] = screenshot()
    psim.SameLine()
    if psim.Button('Turntable (72 frames)'): turntable(72); last[0] = f'72 frames in {args.out}'; fig7_view()
    psim.SameLine()
    if psim.Button('View of Fig. 7'): fig7_view()
    if last[0]: psim.TextUnformatted(f'saved: {last[0]}')
    psim.PopItemWidth()
if args.turntable:
    turntable(args.turntable); print(f'{args.turntable} frames written to {args.out}')
else:
    fig7_view(); ps.set_user_callback(panel); ps.show()
