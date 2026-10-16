"""Interactive 3D viewer on the GPU (OpenGL, Polyscope) for Mizhaev's polyhedron and the surfaces of the sweeps.
Rotate with the left mouse button, pan with the right, zoom with the wheel; every object can be switched on and off
in the panel on the left. The panel 'Mizhaev' has buttons for screenshots (PNG, transparent background) and for a
turntable sequence of frames.
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
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
PAL = np.array([[0x4E, 0x79, 0xA7], [0xF2, 0x8E, 0x2B], [0xE1, 0x57, 0x59], [0x76, 0xB7, 0xB2],
                [0x59, 0xA1, 0x4F], [0xED, 0xC9, 0x48], [0xB0, 0x7A, 0xA1], [0xFF, 0x9D, 0xA7]]) / 255   # nonagon colours of the paper
R = json.load(open(os.path.join(SRC, 'results.json'))); V = np.array(R['vertices'], float)
TRI = np.array(R['triangles']) - 1; FACES = [[x - 1 for x in f] for f in R['faces']]
tri_face = np.array([next(k for k, f in enumerate(FACES) if set(t) <= set(f)) for t in TRI])
cone = np.degrees(2 * np.pi - np.array(R['original_curvature']))             # cone angle at each vertex of P
edges = np.array(sorted({tuple(sorted((f[i], f[(i + 1) % 9]))) for f in FACES for i in range(9)}))

ps.init(args.backend) if args.backend else ps.init()
ps.set_program_name('Mizhaev polyhedron'); ps.set_up_dir('z_up'); ps.set_ground_plane_mode('shadow_only')
ps.set_SSAA_factor(args.ssaa); ps.set_transparency_mode('pretty')
m = ps.register_surface_mesh('P (56 triangles of K)', V, TRI, smooth_shade=False, edge_width=0.0)
m.add_color_quantity('nonagons', PAL[tri_face], defined_on='faces', enabled=True)
ps.register_curve_network('edges of P', V, edges, radius=0.0025, color=(0.11, 0.15, 0.2))
pc = ps.register_point_cloud('cone points', V, radius=0.008)
pc.add_scalar_quantity('cone angle (degrees)', cone, cmap='coolwarm', vminmax=(0, 720), enabled=True)
ps.register_curve_network('symmetry axis', np.array([[0, 0, -330.], [0, 0, 330.]]), np.array([[0, 1]]),
                          radius=0.0012, color=(0.5, 0.5, 0.5), enabled=False)
for k, f in enumerate(args.surfaces):                                       # surfaces from the sweeps
    d = json.load(open(f)); X = np.array(d['vertices'], float); F = np.array(d['faces']) - 1; G = np.array(d['nonagon']) - 1
    if args.side_by_side: X = X + np.array([700.0 * (k + 1), 0, 0])
    s = ps.register_surface_mesh(os.path.splitext(os.path.basename(f))[0], X, F, smooth_shade=True, edge_width=0.0,
                                 enabled=args.side_by_side or k == 0)
    s.add_color_quantity('nonagons', PAL[G], defined_on='faces', enabled=True)
    if not args.side_by_side: m.set_transparency(0.35)                       # P shows through the overlaid surface
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
    if psim.Button('Turntable (72 frames)'): turntable(72); last[0] = f'72 frames in {args.out}'
    if last[0]: psim.TextUnformatted(f'saved: {last[0]}')
    psim.PopItemWidth()
if args.turntable:
    turntable(args.turntable); print(f'{args.turntable} frames written to {args.out}')
else:
    ps.set_user_callback(panel); ps.show()
