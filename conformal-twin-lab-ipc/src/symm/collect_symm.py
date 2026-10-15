"""Merge the homotopy runs: per-run solution counts, agreement between runs, all solutions of the ORIGINAL
22 equations, real ones classified against P. Usage: python collect_symm.py OUTDIR"""
import json, glob, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from symsys import residuals, jacobian, X_of, P
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'alex')); sys.path.insert(0, os.path.dirname(HERE))
from realize import shape_distance, TRI
from geom_np import self_intersections
out = sys.argv[1]; runs = {}
for f in sorted(glob.glob(os.path.join(out, 'symm_run*_batch*.json'))):
    d = json.load(open(f)); r = runs.setdefault(d['run'], dict(paths=0, finite=0, at_infinity=0, singular_finite=0, failed=0, sols=[], seconds=0.0))
    for k in ('paths', 'finite', 'at_infinity', 'singular_finite', 'failed', 'seconds'): r[k] += d[k]
    z = np.array(d['z_re']) + 1j * np.array(d['z_im'])
    ok = np.array(d['res_randomised']) < 1e-9 if len(z) else np.zeros(0, bool)
    r['sols'] += [z[i] for i in np.where(ok)[0]]
    if d.get('singular_finite_re'):
        r.setdefault('sing', []).extend(list(np.array(d['singular_finite_re']) + 1j * np.array(d['singular_finite_im'])))
def keymap(S):
    # rounded key ONLY for matching solutions across runs; the full-precision value is kept for all checks
    return {tuple(np.round(np.r_[s.real, s.imag], 6)): s for s in S}
keys, full = {}, {}
for k, r in sorted(runs.items()):
    m = keymap(r['sols']); keys[k] = set(m); full.update(m)
    print(f"run {k}: {r['paths']:,} paths, {r['finite']:,} finite end points ({len(keys[k]):,} distinct), {r['at_infinity']:,} at infinity, "
          f"{r['singular_finite']} singular finite, {r['failed']} failed mid-path, {r['seconds']/3600:.1f} CPU h")
U = set().union(*keys.values()) if keys else set()
print(f'union over runs: {len(U):,} distinct finite solutions of the randomised system; '
      + ', '.join(f'run {k} misses {len(U - s)}' for k, s in keys.items()))
Z = np.array([full[k] for k in U]) if U else np.zeros((0, 17), complex)      # full precision, not the rounded key
orig = np.abs(residuals(Z)).max(1) < 1e-7 if len(Z) else np.zeros(0, bool)
real = orig & (np.abs(Z.imag).max(1) < 1e-7) if len(Z) else orig
print(f'solutions of the ORIGINAL 22 equations: {orig.sum()} (complex), of which real: {real.sum()}')
S = np.array([s for r in runs.values() for s in r.get('sing', [])])
if len(S):
    ro = np.abs(residuals(S)).max(1); near = (ro < 1e-2) & (np.abs(S.imag).max(1) < 1e-2)
    print(f'singular finite end points: {len(S)}; approximately on the original system and nearly real: {near.sum()} (inspect if > 0)')
classes = []
for z in Z[real].real:
    X = X_of(z); dP = shape_distance(X, P)
    if not any(shape_distance(X, c['X']) < 1e-6 for c in classes):
        classes.append(dict(X=X, dist_to_P=dP, selfint=self_intersections(X, TRI)))
for c in classes:
    print(f"  real realisation class: RMS distance to P {c['dist_to_P']:.3e}, self-intersections {c['selfint']}")
json.dump(dict(runs={k: {kk: v for kk, v in r.items() if kk not in ('sols', 'sing')} for k, r in runs.items()}, union=len(U),
               original_complex=int(orig.sum()), original_real=int(real.sum()),
               real_classes=[dict(X=c['X'].tolist(), dist_to_P=float(c['dist_to_P']), self_intersections=int(c['selfint'])) for c in classes]),
          open(os.path.join(out, 'symm_summary.json'), 'w'), indent=1)
