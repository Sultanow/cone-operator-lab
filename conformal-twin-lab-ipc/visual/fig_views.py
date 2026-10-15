"""Figure 6 of the paper: three views of one object.
Usage: python visual/fig_views.py WEAK.json STRONG.json   (exports of scripts/export_obj.py or .npz checkpoints)
 P itself and two surfaces in R^3 that are discretely conformal to P
(hence to h_P), with the 24 cone points of P marked. Input: npz checkpoints of --target P runs."""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(os.path.dirname(HERE), 'src'); sys.path.insert(0, SRC)
from fine_mesh import build
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8.5})
R = json.load(open(f'{SRC}/results.json')); D = json.load(open(f'{SRC}/development.json')); VP = np.array(R['vertices'], float)
theta = np.degrees(2 * np.pi - np.array(R['original_curvature']))
pal = np.array([[0x4E,0x79,0xA7],[0xF2,0x8E,0x2B],[0xE1,0x57,0x59],[0x76,0xB7,0xB2],[0x59,0xA1,0x4F],[0xED,0xC9,0x48],[0xB0,0x7A,0xA1],[0xFF,0x9D,0xA7]]) / 255
def rot(elev, azim):
    a, e = np.radians(azim), np.radians(elev)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]); Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Rz
def render(X, F, fcol, marks, elev=22, azim=40, N=700, span=335):
    Pm = X @ rot(elev, azim).T; sx, sy, dep = Pm[:, 0], Pm[:, 2], -Pm[:, 1]
    px = (sx / span + 1) * N / 2; py = N - (sy / span + 1) * N / 2
    img = np.ones((N, N, 4)); img[..., 3] = 0; zb = np.full((N, N), np.inf); L = np.array([0.35, 0.8, 0.5]); L /= np.linalg.norm(L)
    for n, (i, j, k) in enumerate(F):
        nr = np.cross(Pm[j] - Pm[i], Pm[k] - Pm[i]); nn = np.linalg.norm(nr)
        if nn == 0: continue
        sh = 0.62 + 0.38 * abs((nr / nn) @ np.array([L[0], -L[2], L[1]])); col = fcol[n] * sh
        xs, ys = px[[i, j, k]], py[[i, j, k]]
        x0, x1 = max(0, int(xs.min())), min(N - 1, int(np.ceil(xs.max()))); y0, y1 = max(0, int(ys.min())), min(N - 1, int(np.ceil(ys.max())))
        if x1 < x0 or y1 < y0: continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + .5, np.arange(y0, y1 + 1) + .5)
        A = np.array([[xs[0], xs[1], xs[2]], [ys[0], ys[1], ys[2]], [1, 1, 1]])
        if abs(np.linalg.det(A)) < 1e-9: continue
        bc = np.linalg.solve(A, np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])); ins = (bc >= -1e-9).all(0)
        d = (bc * dep[[i, j, k]][:, None]).sum(0); yy, xx = gy.ravel().astype(int), gx.ravel().astype(int); m = ins & (d < zb[yy, xx])
        zb[yy[m], xx[m]] = d[m]; img[yy[m], xx[m], :3] = col; img[yy[m], xx[m], 3] = 1
    vis = []
    for v in marks:   # cone points: visible if not hidden behind the surface
        xi, yi = int(px[v]), int(py[v])
        ok = 0 <= xi < N and 0 <= yi < N and dep[v] <= np.min(zb[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3]) + 0.02 * span
        vis.append((px[v], py[v], ok))
    return img, vis
def panel(ax, X, F, face_tri, marks, title):
    img, vis = render(X, F, pal[np.array(D['face_of'])[face_tri]], marks)
    ax.imshow(img); ax.axis('off'); ax.set_title(title, fontsize=8.5)
    for (x, y, ok), v in zip(vis, range(24)):
        if ok: ax.scatter([x], [y], s=14, c=[plt.get_cmap('coolwarm')(theta[v] / 720)], edgecolors='#1b2733', linewidths=.5, zorder=3)
if __name__ == '__main__':
    files = sys.argv[1:3]; labels = ['weakly smoothed', 'strongly smoothed']
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.75), dpi=240)
    M0 = build(0, SRC)
    marks0 = [int(np.argmin(np.linalg.norm(M0['XP'] - VP[i], axis=1))) for i in range(24)]
    panel(axs[0], M0['XP'], M0['F'], M0['face_tri'], marks0, 'the polyhedron $P$')
    for ax, f, lab in zip(axs[1:], files, labels):
        if f.endswith('.json'):     # export of scripts/export_obj.py: vertices, faces (1-based), nonagon per face
            d = json.load(open(f)); X = np.array(d['vertices']); Fc = np.array(d['faces']) - 1
            M = build({24: 0, 108: 1, 444: 2, 1788: 3, 7164: 4}[len(X)], SRC)
        else:
            Z = np.load(f); X, Fc = Z['X'], Z['F']; M = build(int(Z['level']), SRC)
        assert np.array_equal(Fc, M['F'])
        marks = [int(np.argmin(np.linalg.norm(M['XP'] - VP[i], axis=1))) for i in range(24)]
        panel(ax, X, Fc, M['face_tri'], marks, lab)
    fig.subplots_adjust(wspace=0.02, left=0, right=1, top=.88, bottom=0)
    fig.savefig('fig_views.pdf', bbox_inches='tight', pad_inches=.02, dpi=300); fig.savefig('fig_views.png', bbox_inches='tight', pad_inches=.02)
    print('written fig_views.pdf')
