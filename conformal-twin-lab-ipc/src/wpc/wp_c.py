"""Work package C: the polyhedron's own hyperbolic structure h_P versus the circle-packing twin h_cp.
Uniformises P discretely on levels 0..3 and compares lengths of closed geodesics in the same free
homotopy classes (words of length <= 3 in the coarse dual loops).  Usage: python src/wpc/wp_c.py OUTDIR"""
import os, sys, json, itertools, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC); sys.path.insert(0, HERE)
from fine_mesh import build
from holonomy import Developed, length_of
from loops import coarse_loops, realise
from uniformize import uniformize
out = sys.argv[1] if len(sys.argv) > 1 else '.'; os.makedirs(out, exist_ok=True)
loops, _ = coarse_loops(SRC)
def loop_mats(M, L):
    Ld = {frozenset(map(int, e)): l for e, l in zip(M['edges'], L)}
    dev = Developed([tuple(map(int, f)) for f in M['F']], lambda i, j: Ld[frozenset((i, j))])
    fadj = {f: [] for f in range(len(M['F']))}
    for fs in dev.edge_faces.values(): fadj[fs[0]].append(fs[1]); fadj[fs[1]].append(fs[0])
    return [dev.holonomy(realise(lp, M, fadj)) for lp in loops]
M0 = build(0, SRC); Hcp = loop_mats(M0, M0['lenH'])
nontriv = [k for k, H in enumerate(Hcp) if length_of(H) > 1e-3]
letters = [(k, 1) for k in nontriv] + [(k, -1) for k in nontriv]
words = [w for n in (1, 2, 3) for w in itertools.product(letters, repeat=n)
         if not any(w[i][0] == w[i + 1][0] and w[i][1] == -w[i + 1][1] for i in range(n - 1))]
def word_lengths(H):
    Hs = {(k, 1): H[k] for k in nontriv}; Hs.update({(k, -1): np.linalg.inv(H[k]) for k in nontriv})
    res = []
    for w in words:
        A = np.eye(3)
        for l in w: A = A @ Hs[l]
        res.append(length_of(A))
    return np.array(res)
lcp = word_lengths(Hcp); keep = lcp > 1e-3; report = dict(n_curves=int(keep.sum()), levels={})
print(f'{len(nontriv)} nontrivial coarse loops, {keep.sum()} test curves')
for lev in (0, 1, 2, 3):
    M = build(lev, SRC); U = uniformize(M, verbose=False)
    lP = word_lengths(loop_mats(M, U['L'])); r = lP[keep] / lcp[keep]
    report['levels'][lev] = dict(residual=U['residual'], systole_P=float(lP[keep].min()), systole_cp=float(lcp[keep].min()),
                                 ratio_min=float(r.min()), ratio_max=float(r.max()),
                                 thurston_lower_bounds=[float(np.log(r.max())), float(np.log(1 / r.min()))])
    print(f'level {lev}: uniformisation residual {U["residual"]:.1e}; ratio h_P/h_cp in [{r.min():.4f}, {r.max():.4f}]; '
          f'Thurston distance >= {max(np.log(r.max()), np.log(1 / r.min())):.3f}')
json.dump(report, open(os.path.join(out, 'wp_c_report.json'), 'w'), indent=1)
np.savez(os.path.join(out, 'length_compare.npz'), lcp=lcp, keep=keep, P3=lP)   # level-3 lengths, input of visual/fig_lengths.py
