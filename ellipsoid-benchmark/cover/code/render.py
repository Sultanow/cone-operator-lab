"""Cover rendering for 'Hearing Ellipsoids' (AppliedMath 6(11):179).
Ray-casts a cut-away triaxial ellipsoid Omega(1,1.5,2.3): cut faces show a computed
Dirichlet eigenfunction u, the outer shell shows its boundary flux du/dn, and the
background rings sit at the computed wave numbers sqrt(lambda_k).
No generative image model is involved; every pixel follows from the formulas below."""
import numpy as np, argparse
from scipy import ndimage
from PIL import Image

A, B, C = 1.0, 1.5, 2.3

def full_field(npz, m):
    d = np.load(npz); ins = d["inside"]; F = np.zeros(ins.shape); F[ins] = d["vecs"][:, m]
    F = F / np.abs(F).max()
    # all-even symmetry class -> mirror into the full box
    F = np.concatenate([F[::-1], F], 0); F = np.concatenate([F[:, ::-1], F], 1); F = np.concatenate([F[:, :, ::-1], F], 2)
    h = float(d["h"]); n = np.array(ins.shape)
    coef = ndimage.spline_filter(F, order=3, mode="constant")
    def sample(P):  # P: (N,3) world coords
        g = (P / h + n - 0.5).T
        return ndimage.map_coordinates(coef, g, order=3, mode="constant", prefilter=False)
    return sample, float(d["vals"][m]), h

def cmap_div(v):
    """Vivid diverging palette: deep blue - cyan - ivory - amber - magenta."""
    stops = np.array([-1, -0.55, -0.2, 0, 0.2, 0.55, 1.0])
    cols = np.array([[0.10, 0.18, 0.75], [0.05, 0.55, 0.98], [0.45, 0.90, 1.00], [1.0, 0.97, 0.90],
                     [1.00, 0.80, 0.30], [1.00, 0.42, 0.20], [0.88, 0.10, 0.55]])
    v = np.clip(v, -1, 1)
    return np.stack([np.interp(v, stops, cols[:, i]) for i in range(3)], -1)

def render(W, H, sample, cam, look, fov, ss, rows=192, ylo=0, yhi=None):
    Wp, Hp = W * ss, H * ss
    fwd = look - cam; fwd /= np.linalg.norm(fwd)
    upw = np.array([0.0, 0.0, 1.0]); right = np.cross(fwd, upw); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    tanf = np.tan(np.radians(fov) / 2); aspect = Wp / Hp
    yhi = H if yhi is None else yhi
    img = np.zeros(((yhi - ylo) * ss, Wp, 3)); alpha = np.zeros(((yhi - ylo) * ss, Wp))
    s = np.array([A, B, C])
    L1 = np.array([0.55, 0.25, 0.80]); L1 /= np.linalg.norm(L1)      # key light
    L2 = np.array([-0.6, 0.9, 0.2]); L2 /= np.linalg.norm(L2)        # fill light
    for r0 in range(ylo * ss, yhi * ss, rows):
        r1 = min(yhi * ss, r0 + rows)
        yy, xx = np.mgrid[r0:r1, 0:Wp]
        px = ((xx + 0.5) / Wp * 2 - 1) * tanf * aspect
        py = (1 - (yy + 0.5) / Hp * 2) * tanf
        D = fwd + px[..., None] * right + py[..., None] * up
        D = (D / np.linalg.norm(D, axis=-1, keepdims=True)).reshape(-1, 3)
        O = np.broadcast_to(cam, D.shape)
        o2, d2 = O / s, D / s
        qa = (d2 * d2).sum(1); qb = 2 * (o2 * d2).sum(1); qc = (o2 * o2).sum(1) - 1
        disc = qb * qb - 4 * qa * qc
        hitE = disc > 0
        sq = np.sqrt(np.where(hitE, disc, 0))
        tin = (-qb - sq) / (2 * qa); tout = (-qb + sq) / (2 * qa)
        INF = np.inf
        best = np.full(len(D), INF); kind = np.full(len(D), -1)
        Pin = O + tin[:, None] * D
        inO = (Pin > 0).all(1)
        ok = hitE & ~inO
        best[ok] = tin[ok]; kind[ok] = 3
        for ax in range(3):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = -O[:, ax] / D[:, ax]
            Q = O + t[:, None] * D
            oth = [i for i in range(3) if i != ax]
            v = hitE & (t >= tin - 1e-9) & (t <= tout + 1e-9) & (Q[:, oth[0]] >= 0) & (Q[:, oth[1]] >= 0) \
                & (((Q / s) ** 2).sum(1) <= 1) & (t < best)
            best[v] = t[v]; kind[v] = ax
        hit = kind >= 0
        col = np.zeros((len(D), 3)); a = np.zeros(len(D))
        if hit.any():
            P = O[hit] + best[hit, None] * D[hit]; k = kind[hit]; Dv = D[hit]
            N = np.zeros_like(P)
            sh = k == 3
            N[sh] = P[sh] / s**2; N[sh] /= np.linalg.norm(N[sh], axis=1, keepdims=True)
            for ax in range(3):
                N[k == ax, ax] = 1.0
            val = np.zeros(len(P))
            cutm = ~sh
            val[cutm] = sample(P[cutm])
            if sh.any():
                dlt = 0.06
                g = -sample(P[sh] - dlt * N[sh]) / dlt
                val[sh] = g
            base = np.zeros((len(P), 3))
            base[cutm] = cmap_div(val[cutm] / 0.85)
            gsc = 1.0 / 7.0
            shell = cmap_div(np.tanh(val[sh] * gsc * 1.6))
            pearl = np.array([0.93, 0.95, 1.0])
            base[sh] = 0.72 * shell + 0.28 * pearl
            V = -Dv
            diff = 0.55 * np.clip((N * L1).sum(1), 0, 1) + 0.18 * np.clip((N * L2).sum(1), 0, 1) + 0.38 + 0.12 * sh
            Hh = (L1 + V); Hh /= np.linalg.norm(Hh, axis=1, keepdims=True)
            spec = np.clip((N * Hh).sum(1), 0, 1) ** 60
            fres = (1 - np.clip((N * V).sum(1), 0, 1)) ** 3
            c = base * diff[:, None]
            c[sh] += 0.55 * spec[sh, None] + 0.35 * fres[sh, None] * np.array([0.55, 0.8, 1.0])
            c[cutm] += 0.12 * spec[cutm, None]
            # soft crease darkening where cut faces meet (ambient occlusion feel)
            if cutm.any():
                Pc = P[cutm]; kc = k[cutm]
                dmin = np.full(len(Pc), 9.0)
                for ax in range(3):
                    dd = np.where(kc == ax, 9.0, np.abs(Pc[:, ax]))
                    dmin = np.minimum(dmin, dd)
                ao = 1 - 0.32 * np.exp(-dmin / 0.07)
                c[cutm] *= ao[:, None]
                # bright rim where a cut face meets the ellipsoid surface
                rad = np.sqrt(((Pc / s) ** 2).sum(1))
                c[cutm] += 0.5 * np.exp(-((1 - rad) / 0.012) ** 2)[:, None] * np.array([1.0, 0.95, 0.85])
            col[hit] = np.clip(c, 0, 1); a[hit] = 1
        img[r0 - ylo * ss:r1 - ylo * ss] = col.reshape(r1 - r0, Wp, 3); alpha[r0 - ylo * ss:r1 - ylo * ss] = a.reshape(r1 - r0, Wp)
    if ss > 1:
        hh = yhi - ylo
        img = img.reshape(hh, ss, W, ss, 3).mean((1, 3)); alpha = alpha.reshape(hh, ss, W, ss).mean((1, 3))
    return img, alpha

def background(W, H, cx, cy):
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot((xx - cx) / W, (yy - cy) / W)
    c0 = np.array([0.16, 0.20, 0.52]); c1 = np.array([0.03, 0.04, 0.14])
    t = np.clip(r / 0.85, 0, 1)[..., None] ** 0.9
    return c0 * (1 - t) + c1 * t

def rings(W, H, cx, cy, rx, ry, ang, lev, rho0=1.14, rho1=2.55, width=1.6):
    """Concentric ellipses at the computed wave numbers k = sqrt(lambda)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    X, Y = xx - cx, yy - cy
    ca, sa = np.cos(ang), np.sin(ang)
    u = (X * ca + Y * sa) / rx; v = (-X * sa + Y * ca) / ry
    rho = np.hypot(u, v)
    kk = np.sqrt(lev); kk = (kk - kk[0]) / (kk[-1] - kk[0])
    rk = rho0 + (rho1 - rho0) * kk
    grid = np.linspace(0, rho1 + 0.5, 20000)
    prof = np.zeros_like(grid); pxw = width / min(rx, ry)
    for r, q in zip(rk, kk):
        prof += (1 - 0.55 * q) * np.exp(-0.5 * ((grid - r) / pxw) ** 2)
    prof = 1 - np.exp(-prof * 0.8)  # soft saturation where levels coincide
    I = np.interp(rho, grid, prof)
    fade = np.clip((rho1 + 0.25 - rho) / 1.1, 0, 1) ** 1.7 * np.clip((rho - 1.0) / 0.08, 0, 1)
    tcol = np.clip((rho - rho0) / (rho1 - rho0), 0, 1)
    stops = np.array([0, 0.35, 0.7, 1.0])
    cols = np.array([[0.45, 0.95, 1.0], [0.60, 0.70, 1.0], [0.95, 0.50, 0.95], [1.0, 0.75, 0.45]])
    col = np.stack([np.interp(tcol, stops, cols[:, i]) for i in range(3)], -1)
    return col, (I * fade)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, default=620); ap.add_argument("--H", type=int, default=877)
    ap.add_argument("--ss", type=int, default=1); ap.add_argument("--mode", type=int, default=21)
    ap.add_argument("--npz", default="scan_even.npz"); ap.add_argument("--spec", default="fd_spectrum.npy")
    ap.add_argument("--out", default="proto.png"); ap.add_argument("--dist", type=float, default=13.5); ap.add_argument("--lookz", type=float, default=0.55); ap.add_argument("--layers", action="store_true"); ap.add_argument("--part", default=None); ap.add_argument("--assemble", type=int, default=0)
    a = ap.parse_args()
    sample, lam, h = full_field(a.npz, a.mode)
    dirv = np.array([7.2, 5.6, 4.3]); dirv /= np.linalg.norm(dirv); cam = dirv * a.dist; look = np.array([0.0, 0.0, a.lookz])
    if a.part is not None:
        i, n = map(int, a.part.split("/")); edges = np.linspace(0, a.H, n + 1).astype(int)
        im, al = render(a.W, a.H, sample, cam, look, fov=34, ss=a.ss, ylo=edges[i], yhi=edges[i + 1])
        np.savez(f"part_{i}.npz", img=im.astype(np.float32), alpha=al.astype(np.float32)); raise SystemExit
    if a.assemble:
        n = a.assemble; P = [np.load(f"part_{i}.npz") for i in range(n)]
        img = np.concatenate([p["img"] for p in P]).astype(float); alpha = np.concatenate([p["alpha"] for p in P]).astype(float)
    else:
        img, alpha = render(a.W, a.H, sample, cam, look, fov=34, ss=a.ss)
    ys, xs = np.nonzero(alpha > 0.5)
    cx, cy = xs.mean(), ys.mean()
    cov = np.cov(np.stack([xs - cx, ys - cy]))
    ev, evec = np.linalg.eigh(cov)
    rx, ry = 2 * np.sqrt(ev[1]), 2 * np.sqrt(ev[0]); ang = np.arctan2(evec[1, 1], evec[0, 1])
    bg = background(a.W, a.H, cx, cy)
    lev = np.load(a.spec) if a.spec else None
    if lev is not None:
        rc, ra = rings(a.W, a.H, cx, cy, rx, ry, ang, lev, width=1.1 * a.W / 2480 * 2.2)
        comp = bg * (1 - ra[..., None] * 0.6) + rc * ra[..., None] * 0.6
    else:
        comp = bg
    # soft glow behind the body
    glow = ndimage.gaussian_filter(alpha, a.W / 40)[..., None]
    comp = comp + 0.25 * glow * np.array([0.4, 0.6, 1.0])
    comp = comp * (1 - alpha[..., None]) + img * alpha[..., None]
    Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8)).save(a.out, dpi=(300, 300))
    if a.layers:
        Image.fromarray((np.clip(bg, 0, 1) * 255).astype(np.uint8)).save(a.out.replace(".png", "_layer1_background.png"), dpi=(300, 300))
        if lev is not None:
            rgba = np.dstack([rc, ra * 0.6]); Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA").save(a.out.replace(".png", "_layer2_spectral_rings.png"), dpi=(300, 300))
        rgba = np.dstack([img, alpha]); Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA").save(a.out.replace(".png", "_layer3_ellipsoid.png"), dpi=(300, 300))
    print("lambda =", lam, "silhouette", cx, cy, rx, ry, np.degrees(ang))
