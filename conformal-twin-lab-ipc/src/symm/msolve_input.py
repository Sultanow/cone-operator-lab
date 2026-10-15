"""Exact form of the C4-symmetric realisation problem for msolve (Groebner bases over Q).
Unknowns (17, unscaled coordinates): x_0 = (v1, 0, v2), x_k = (v{3k}, v{3k+1}, v{3k+2}) for k = 1..5.
Equations (22, integer coefficients): |x_k - T^d x_l|^2 - D_kld = 0, one per tau-orbit of edges of K.
If msolve reports a zero-dimensional ideal with exactly four real solutions, every C4-symmetric realisation
on K is one of the four copies of P: an exact, computer-assisted proof of the statement of Proposition 8.
Also writes Singular scripts that compute the reduced Groebner basis over Q with deterministic exact arithmetic
(std and slimgb), whose vdim is the number of complex solutions counted with multiplicity.
Usage: python src/symm/msolve_input.py OUTDIR   -> OUTDIR/symm_system.ms, OUTDIR/symm_{std,slimgb}.sing"""
import os, sys, numpy as np, sympy as sp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from symsys import KEYS, eqs, Tp, z_of, P, S
v = sp.symbols('v1:18'); x = [sp.Matrix([v[0], 0, v[1]])] + [sp.Matrix(v[2 + 3 * k:5 + 3 * k]) for k in range(5)]
T = [sp.Matrix(np.rint(M).astype(int).tolist()) for M in Tp]
polys = []
for (k, l, d) in KEYS:
    diff = x[k] - T[d] * x[l]; polys.append(sp.expand(diff.dot(diff) - eqs[(k, l, d)]))
assert len(polys) == 22 and all(p.is_polynomial(*v) for p in polys)
assert all(all(c.is_integer for c in sp.Poly(p, *v).coeffs()) for p in polys), 'coefficients must be integers'
# check: the (rotated) polyhedron solves the system to rounding precision
z = z_of(P) * S; sub = dict(zip(v, z))
res = max(abs(float(p.subs(sub))) / max(eqs.values()) for p in polys)
print(f'22 polynomials in 17 unknowns, integer coefficients; relative residual of P: {res:.1e}')
out = sys.argv[1] if len(sys.argv) > 1 else '.'; os.makedirs(out, exist_ok=True)
names = ','.join(str(s) for s in v); body = [str(p).replace('**', '^').replace(' ', '') for p in polys]
with open(os.path.join(out, 'symm_system.ms'), 'w') as fh:
    fh.write(names + '\n0\n' + ',\n'.join(body) + '\n')
for alg in ('std', 'slimgb'):
    res = os.path.join(os.path.abspath(out), f'symm_groebner_{alg}.txt')
    with open(os.path.join(out, f'symm_{alg}.sing'), 'w') as fh:
        fh.write(f"""// reduced Groebner basis over Q, deterministic exact arithmetic ({alg})
ring r = 0, ({names}), dp;
ideal I = {', '.join(body)};
option(redSB);
int t0 = timer;
ideal G = {alg}(I);
"dimension of the ideal: " + string(dim(G));
"number of complex solutions with multiplicity (vdim): " + string(vdim(G));
"size of the reduced Groebner basis: " + string(size(G));
"seconds: " + string(timer - t0);
write(":w {res}", G);
quit;
""")
print('written', os.path.join(out, 'symm_system.ms'), 'and the Singular scripts symm_std.sing, symm_slimgb.sing')
