"""Pick the best ribbon-graph layout from the cluster search and redraw it.
Best = fewest twisted bands, then fewest crossings, then the largest smallest angle between bands at a vertex.
Usage: python scripts/pick_ribbon.py $CT_OUT/ribbon  -> $CT_OUT/ribbon/best/fig_ribbon.pdf"""
import os, sys, glob, json, subprocess
root = sys.argv[1]; rows = []
for f in glob.glob(os.path.join(root, 'seed_*', 'fig_ribbon_layout.json')):
    d = json.load(open(f)); rows.append((d['twists'], d['crossings'], -d['min_angle_deg'], d['seed'], f))
rows.sort(); print(f'{len(rows)} layouts; without twists: {sum(r[0] == 0 for r in rows)}')
for t, c, a, sd, f in rows[:5]: print(f'  seed {sd:3d}: {t} twisted bands, {c} crossings, smallest angle {-a:.0f} deg')
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, os.path.join(here, '..', 'visual', 'fig_ribbon.py'), os.path.join(root, 'best'), '--layout', rows[0][4]], check=True)
