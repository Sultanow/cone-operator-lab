"""Evaluate the exact solves of the symmetric system: msolve's real solutions (isolating boxes) are mapped back to
configurations and compared with P; Singular logs report the dimension and the degree (vdim) of the ideal.
Usage: python src/symm/check_algebraic.py OUTDIR"""
import os, re, sys, glob, json, numpy as np
from fractions import Fraction
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(SRC, 'alex')); sys.path.insert(0, SRC)
from symsys import X_of, S, P
from realize import shape_distance, align
from geom_np import self_intersections
out = sys.argv[1]; TRI = np.array(json.load(open(os.path.join(SRC, 'results.json')))['triangles']) - 1
f = os.path.join(out, 'symm_system.out')
if os.path.exists(f):
    s = re.sub(r'(-?\d+)\s*/\s*2\^(\d+)', r'Fraction(\1, 2**\2)', open(f).read().strip().rstrip(':'))
    data = eval(s); dim = data[0]
    print(f'msolve: dimension {dim}' + ('' if dim != 0 else f', {len(data[1][1])} real solutions'))
    if dim == 0:
        for k, sol in enumerate(data[1][1]):
            lo = np.array([float(a) for a, b in sol]); hi = np.array([float(b) for a, b in sol]); X = X_of((lo + hi) / 2 / S)
            print(f'  solution {k + 1}: distance to P after congruence {shape_distance(X, P):.1e}, '
                  f'self-intersections {self_intersections(align(X, P)[1], TRI)}, box width {np.max(hi - lo):.1e}')
for log in sorted(glob.glob(os.path.join(out, 'singular_*.log'))):
    txt = open(log).read(); print(os.path.basename(log) + ':', ' | '.join(l.strip() for l in txt.splitlines() if l.strip()) or 'no result yet')
