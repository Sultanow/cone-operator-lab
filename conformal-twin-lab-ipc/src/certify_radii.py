"""Computer-assisted proof (Krawczyk operator in Arb ball arithmetic) that the
circle-packing equations Theta_i(r) = 2*pi of the triangulation K have a unique
solution in an explicit box, i.e. rigorous enclosures of the 24 radii.

Theorem checked: if K(X) is contained in the interior of X, then F has exactly one
zero in X (Krawczyk 1969, Moore 1977). Variables u_i = log r_i."""
import json, os, sys, numpy as np
from flint import arb, arb_mat, ctx

PREC = int(sys.argv[1]) if len(sys.argv) > 1 else 256
OUT = sys.argv[2] if len(sys.argv) > 2 else 'certified_radii.json'
HERE = os.path.dirname(os.path.abspath(__file__))
ctx.prec = PREC
R = json.load(open(os.path.join(HERE, 'results.json')))
TRI = [tuple(x - 1 for x in t) for t in R['triangles']]
N = 24

def corner(ri, rj, rk):
    """angle at vertex i and its partials w.r.t. (r_i, r_j, r_k); works for floats via arb too"""
    a, b, C = ri + rj, ri + rk, rj + rk
    ca, cb, cC = a.cosh(), b.cosh(), C.cosh()
    sa, sb, sC = a.sinh(), b.sinh(), C.sinh()
    c = (ca * cb - cC) / (sa * sb)
    s = (1 - c * c).sqrt()                      # sin(alpha) > 0
    alpha = c.acos()
    dca = (cC * ca - cb) / (sa * sa * sb)
    dcb = (cC * cb - ca) / (sb * sb * sa)
    dcC = -sC / (sa * sb)
    return alpha, (-(dca + dcb) / s, -(dca + dcC) / s, -(dcb + dcC) / s)

def F_and_J(u):
    r = [ui.exp() for ui in u]
    F = [arb(0)] * N
    J = [[arb(0)] * N for _ in range(N)]
    for (i, j, k) in TRI:
        for (p, q, w) in ((i, j, k), (j, k, i), (k, i, j)):
            al, (dp, dq, dw) = corner(r[p], r[q], r[w])
            F[p] = F[p] + al
            J[p][p] = J[p][p] + dp * r[p]
            J[p][q] = J[p][q] + dq * r[q]
            J[p][w] = J[p][w] + dw * r[w]
    twopi = 2 * arb.pi()
    return [f - twopi for f in F], J

# ---- 1. high-precision Newton for the midpoint (start from double-precision radii)
u = [arb(float(np.log(x))) for x in R['radii']]
for it in range(8):
    F, J = F_and_J(u)
    Jm = arb_mat([[J[a][b].mid() for b in range(N)] for a in range(N)])
    step = Jm.solve(arb_mat([[f.mid()] for f in F]))
    u = [(u[a] - step[a, 0]).mid() for a in range(N)]
    res = max(abs(float(f.mid())) for f in F)
print(f'precision {PREC} bits, Newton residual before last step {res:.3e}')

# ---- 2. Krawczyk test on X = u~ +/- rho
F0, J0 = F_and_J(u)
Y = arb_mat([[J0[a][b].mid() for b in range(N)] for a in range(N)]).inv()
Y = arb_mat([[Y[a, b].mid() for b in range(N)] for a in range(N)])      # exact floating matrix
for rho_exp in (-60, -50, -40, -30, -20):
    rho = arb(2) ** (int(rho_exp * 3.3219))
    X = [arb(ui.mid(), rho) for ui in u]
    _, JX = F_and_J(X)
    JXm = arb_mat(JX)
    I = arb_mat([[1 if a == b else 0 for b in range(N)] for a in range(N)])
    Fm = arb_mat([[f] for f in F0])
    D = arb_mat([[arb(0, rho)] for _ in range(N)])
    Kx = arb_mat([[ui.mid()] for ui in u]) - Y * Fm + (I - Y * JXm) * D
    inside = all(X[a].contains_interior(Kx[a, 0]) for a in range(N))   # rigorous ball containment
    print(f'box radius 1e{rho_exp}: Krawczyk inclusion {"HOLDS" if inside else "fails"}')
    if inside:
        break
# ---- 3. report rigorous radius enclosures (exp of the box)
rad = [arb(ui.mid(), rho).exp() for ui in u]
out = dict(theorem='unique solution of Theta_i(r)=2pi in the box (Krawczyk inclusion)', precision_bits=PREC, box_radius_log=f'1e{rho_exp}',
           radii=[r.str(40, radius=True) for r in rad],
           radii_midpoints_70_digits=[r.mid().str(70) for r in rad], radii_ball_radius=[float(r.rad()) for r in rad])
out['krawczyk_holds'] = bool(inside)
json.dump(out, open(OUT, 'w'), indent=1)
for i in (4, 17, 0, 1, 6, 5):
    print(f'r_{i+1:<2d} in {rad[i].str(30, radius=True)}')
# symmetry check on the enclosures
perm = [p - 1 for p in R['symmetry_perm']]
print('enclosures of tau-related radii overlap:', all(rad[i].overlaps(rad[perm[i]]) for i in range(N)))
