"""Search for further realisations of the intrinsic metric of P (same geodesic triangulation K):
many random starts, Levenberg-Marquardt on |X_i-X_j|^2 = D_ij (84 edges), classification modulo
congruence and the relabelling symmetry tau. Usage:
  python search.py --seed 1 --n 20000 --out DIR [--modes uniform,noise,pop]"""
import argparse, json, os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src/ for geom_np
import realize as RZ
from realize import P, shape_distance, planarity
from geom_np import self_intersections
ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--n', type=int, default=2000)
ap.add_argument('--modes', type=str, default='uniform,noise,pop'); ap.add_argument('--out', type=str, default='out')
ap.add_argument('--retriangulate', action='store_true', help='fold the nonagons along a random other set of diagonals')
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(args.seed); modes = args.modes.split(',')
TRI = RZ.TRI
if args.retriangulate:
    TRI, edges = RZ.random_triangulation(rng); RZ.use_edges(edges)
    print(f'random triangulation: {len(TRI)} triangles, {len(edges)} edges, '
          f'{len({tuple(map(int, e)) for e in edges} - {tuple(sorted(e)) for e in (np.array(RZ.R["edges"]) - 1).tolist()})} diagonals differ from K', flush=True)
E = RZ.E; SCALE = RZ.SCALE; solve = RZ.solve
NB = [sorted(set(E[E[:, 0] == v, 1]) | set(E[E[:, 1] == v, 0])) for v in range(24)]
def start(mode):
    if mode == 'uniform':                                    # anywhere in a ball of P's size
        X = rng.normal(size=(24, 3)); return 300 * X / np.linalg.norm(X, axis=1, keepdims=True) * rng.random((24, 1)) ** (1 / 3)
    if mode == 'noise':                                      # P with large noise
        return P + rng.choice([30, 100, 300]) * rng.normal(size=(24, 3))
    if mode == 'pop':                                        # reflect random vertex sets through local or global planes
        X = P.copy(); k = rng.integers(1, 9)
        for v in rng.choice(24, k, replace=False):
            Q = X[NB[v]]; c = Q.mean(0); n = np.linalg.svd(Q - c)[2][-1]
            X[v] = X[v] - 2 * ((X[v] - c) @ n) * n
        if rng.random() < 0.5:                               # sometimes also a global half-space reflection
            n = rng.normal(size=3); n /= np.linalg.norm(n); side = (X @ n) > rng.normal() * 50
            X[side] = X[side] - 2 * np.outer(X[side] @ n, n)
        return X + 5 * rng.normal(size=X.shape)
found, stats = [], dict(starts=0, converged=0, trivial=0, new=0, seconds=0.0, by_mode={m: [0, 0, 0] for m in modes})
t0 = time.time()
for it in range(args.n):
    m = modes[it % len(modes)]; X0 = start(m); X, rr = solve(X0); stats['starts'] += 1; stats['by_mode'][m][0] += 1
    if rr > 1e-11: continue
    stats['converged'] += 1; stats['by_mode'][m][1] += 1
    dP = shape_distance(X, P)
    if dP < 1e-6 * SCALE: stats['trivial'] += 1; continue
    if any(shape_distance(X, np.array(f['X'])) < 1e-6 * SCALE for f in found): continue
    si = self_intersections(X, TRI)
    found.append(dict(X=X.tolist(), residual=rr, dist_to_P=float(dP), planarity=float(planarity(X)), self_intersections=int(si), mode=m, start_index=it))
    stats['new'] += 1; stats['by_mode'][m][2] += 1
    print(f'  NEW realisation #{len(found)} (start {it}, mode {m}): residual {rr:.1e}, RMS distance to P {dP:.2f}, '
          f'planarity {planarity(X):.2f}, self-intersections {si}', flush=True)
stats['seconds'] = round(time.time() - t0, 1)
json.dump(dict(args=vars(args), stats=stats, found=found, triangles=np.asarray(TRI).tolist()), open(f'{args.out}/search_seed{args.seed}.json', 'w'), indent=1)
print(json.dumps(stats))
