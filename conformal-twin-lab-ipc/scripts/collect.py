"""Aggregate all runs in an output directory: CSV of final metrics per level and a ranking.
Usage: python scripts/collect.py OUTDIR"""
import json, glob, os, sys, csv
out = sys.argv[1]; rows = []
for f in sorted(glob.glob(os.path.join(out, '*.json'))):
    S = json.load(open(f)); a = S['args']
    for lev, e in sorted(S['levels'].items(), key=lambda kv: int(kv[0])):
        m = e['final']
        rows.append(dict(run=os.path.basename(f)[:-5], level=int(lev), w_fit=a['w_fit'], w_will=a['w_will'], barrier=a['barrier'], solver=a.get('solver', 'lbfgs'),
                         seed=a['seed'], noise=a['noise'], V=m['V'], conf_rms=m['conf_rms'], conf_max=m['conf_max'],
                         dist_rms=m['dist_rms'], dist_max=m['dist_max'], W_over_4pi=m['willmore_over_4pi'],
                         self_intersections=m['self_intersections'], sym_err=m['sym_err'],
                         coarse_move=e.get('move_of_coarse_vertices_max', ''), seconds=m['seconds']))
if not rows: sys.exit('no results found in ' + out)
with open(os.path.join(out, 'summary.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
deep = max(r['level'] for r in rows); top = [r for r in rows if r['level'] == deep]
emb = sorted([r for r in top if r['self_intersections'] == 0], key=lambda r: (r['conf_rms'], r['dist_rms']))
imm = sorted([r for r in top if r['self_intersections'] > 0], key=lambda r: (r['conf_rms'], r['dist_rms']))
print(f'{len(rows)} level results from {len(set(r["run"] for r in rows))} runs; deepest level {deep}: {len(top)} runs, {len(emb)} embedded')
fmt = '{run:<44} conf_rms={conf_rms:.2e} dist_rms={dist_rms:7.2f} dist_max={dist_max:7.2f} W/4pi={W_over_4pi:6.3f} selfint={self_intersections:<4} move={coarse_move}'
print('\nEMBEDDED, best conformal residual first:')
for r in emb[:15]: print(fmt.format(**r))
print('\nIMMERSED (self-intersecting), best conformal residual first:')
for r in imm[:8]: print(fmt.format(**r))
# Pareto front among embedded: smaller conf_rms and smaller dist_rms
front = [r for r in emb if not any(o['conf_rms'] <= r['conf_rms'] and o['dist_rms'] <= r['dist_rms'] and o is not r and
                                   (o['conf_rms'] < r['conf_rms'] or o['dist_rms'] < r['dist_rms']) for o in emb)]
print('\nPARETO FRONT (embedded; conformal residual vs. distance to P):')
for r in sorted(front, key=lambda r: r['dist_rms']): print(fmt.format(**r))
