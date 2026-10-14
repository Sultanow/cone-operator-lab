"""Merge the Alexandrov-type search results: totals and all distinct non-trivial realisations.
Usage: python scripts/collect_alex.py OUTDIR"""
import json, glob, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'alex'))
from realize import shape_distance, SCALE
out = sys.argv[1]; files = sorted(glob.glob(os.path.join(out, 'search_seed*.json')))
tot = {'K': dict(tasks=0, starts=0, converged=0, trivial=0, new=0, seconds=0.0), 'retriangulated': dict(tasks=0, starts=0, converged=0, trivial=0, new=0, seconds=0.0)}
allnew = []
for f in files:
    S = json.load(open(f)); kind = 'retriangulated' if S['args'].get('retriangulate') else 'K'; t = tot[kind]; t['tasks'] += 1
    for k in ('starts', 'converged', 'trivial', 'new', 'seconds'): t[k] += S['stats'][k]
    for r in S['found']: r['file'] = os.path.basename(f); r['kind'] = kind; allnew.append(r)
uniq = []
for r in allnew:
    X = np.array(r['X'])
    if not any(shape_distance(X, np.array(u['X'])) < 1e-6 * SCALE for u in uniq): uniq.append(r)
for kind, t in tot.items():
    if t['tasks']:
        print(f"{kind:15s}: {t['tasks']} tasks, {t['starts']:,} starts, {t['converged']:,} exact solutions, {t['trivial']:,} congruent to P, "
              f"{t['new']} new (per task), {t['seconds'] / 3600:.1f} CPU hours")
print(f'distinct non-trivial realisations overall: {len(uniq)}')
for k, u in enumerate(uniq):
    print(f"  #{k + 1}: {u['kind']}, RMS distance to P {u['dist_to_P']:.2f}, planarity {u['planarity']:.3f}, "
          f"self-intersections {u['self_intersections']}, residual {u['residual']:.1e}  ({u['file']}, start {u['start_index']})")
json.dump(dict(totals=tot, distinct=uniq), open(os.path.join(out, 'alex_summary.json'), 'w'), indent=1)
