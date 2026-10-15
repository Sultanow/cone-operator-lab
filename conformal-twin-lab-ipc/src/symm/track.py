"""Vectorised total-degree homotopy continuation for square polynomial systems of degree 2.
H(z,t) = (1-t) gamma G(z) + t F(z),  G_i = z_i^2 - 1,  start points {+-1}^n.
RK4 predictor, Newton corrector with a contraction test, adaptive step per path.
Returns end points at t = 1 with a status: 0 finite, 1 diverged (to infinity), 2 failed."""
import numpy as np

def _solve(M, b):
    """batched solve that never aborts: matrices that are exactly singular (or contain NaN) give NaN,
    so the affected path is rejected and its step halved instead of crashing the whole batch"""
    try:
        return np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        out = np.full(np.broadcast_shapes(M.shape[:-1], b.shape[:-2] + (M.shape[-1],)) + b.shape[-1:], np.nan, dtype=complex)
        for i in range(M.shape[0]):
            try: out[i] = np.linalg.solve(M[i], b[i])
            except np.linalg.LinAlgError: pass
        return out
def track(F, JF, n, starts, gamma, max_steps=40000, tol=1e-11, h0=0.02, hmax=0.05, hmin=1e-13, zmax=1e7):
    B = len(starts); z = starts.astype(complex).copy(); t = np.zeros(B); h = np.full(B, h0)
    status = np.full(B, -1); streak = np.zeros(B, int); steps = np.zeros(B, int)
    def Hz(z, t):
        return (1 - t)[:, None, None] * gamma * 2 * np.einsum('bi,ij->bij', z, np.eye(n)) + t[:, None, None] * JF(z)
    def Ht(z):
        return -gamma * (z * z - 1) + F(z)
    def H(z, t): return (1 - t)[:, None] * gamma * (z * z - 1) + t[:, None] * F(z)
    def vel(z, t): return -_solve(Hz(z, t), Ht(z)[..., None])[..., 0]
    for it in range(max_steps):
        A = np.where(status < 0)[0]
        if len(A) == 0: break
        za, ta, ha = z[A], t[A], np.minimum(h[A], 1 - t[A])
        with np.errstate(all='ignore'):
            k1 = vel(za, ta); k2 = vel(za + 0.5 * ha[:, None] * k1, ta + 0.5 * ha); k3 = vel(za + 0.5 * ha[:, None] * k2, ta + 0.5 * ha)
            k4 = vel(za + ha[:, None] * k3, ta + ha); zp = za + ha[:, None] / 6 * (k1 + 2 * k2 + 2 * k3 + k4); tn = ta + ha
            ok = np.isfinite(zp).all(1); d_prev = None; conv = ok.copy()
            for c in range(3):
                dz = _solve(Hz(zp, tn), H(zp, tn)[..., None])[..., 0]; zp = zp - dz
                nd = np.linalg.norm(dz, axis=1) / (1 + np.linalg.norm(zp, axis=1))
                if d_prev is not None: conv &= (nd <= 0.5 * d_prev + 1e-14)
                d_prev = nd
            conv &= np.isfinite(zp).all(1) & (d_prev < tol * 100)
        acc = A[conv]; rej = A[~conv]
        z[acc] = zp[conv]; t[acc] = tn[conv]; steps[acc] += 1; streak[acc] += 1
        grow = acc[streak[acc] >= 5]; h[grow] = np.minimum(2 * h[grow], hmax); streak[grow] = 0
        h[rej] /= 2; streak[rej] = 0
        done = acc[t[acc] >= 1 - 1e-15]; status[done] = 0
        big = A[np.linalg.norm(np.where(np.isfinite(z[A]), z[A], 1e300), axis=1) > zmax]; status[big] = 1
        status[rej[h[rej] < hmin]] = 2
    status[status < 0] = 2
    return z, status, steps
def newton(F, JF, z, iters=20):
    for _ in range(iters):
        with np.errstate(all='ignore'):
            dz = _solve(JF(z), F(z)[..., None])[..., 0]; z = z - dz
    return z

def track_projective(Fh, JFh, n, starts_affine, gamma, v, max_steps=40000, tol=1e-11, h0=0.02, hmax=0.05, hmin=1e-13):
    """Same predictor-corrector in projective space: w = (w0, z) in C^{n+1}, homogenised target Fh(w),
    start system z_i^2 - w0^2, random affine patch v . w = 1.  Paths stay bounded; w0 -> 0 means infinity."""
    B = len(starts_affine); W = np.c_[np.ones(B), starts_affine].astype(complex); W = W / (W @ v)[:, None]
    t = np.zeros(B); h = np.full(B, h0); status = np.full(B, -1); streak = np.zeros(B, int); steps = np.zeros(B, int)
    def G(W): return W[:, 1:] ** 2 - W[:, [0]] ** 2
    def JG(W):
        J = np.zeros((len(W), n, n + 1), complex); J[:, :, 0] = -2 * W[:, [0]]; J[:, np.arange(n), np.arange(n) + 1] = 2 * W[:, 1:]; return J
    def Hfull(W, t): return np.c_[(1 - t)[:, None] * gamma * G(W) + t[:, None] * Fh(W), W @ v - 1]
    def HW(W, t):
        top = (1 - t)[:, None, None] * gamma * JG(W) + t[:, None, None] * JFh(W)
        return np.concatenate([top, np.broadcast_to(v, (len(W), 1, n + 1))], 1)
    def Ht(W): return np.c_[-gamma * G(W) + Fh(W), np.zeros(len(W))]
    def vel(W, t): return -_solve(HW(W, t), Ht(W)[..., None])[..., 0]
    for it in range(max_steps):
        A = np.where(status < 0)[0]
        if len(A) == 0: break
        Wa, ta, ha = W[A], t[A], np.minimum(h[A], 1 - t[A])
        with np.errstate(all='ignore'):
            k1 = vel(Wa, ta); k2 = vel(Wa + 0.5 * ha[:, None] * k1, ta + 0.5 * ha); k3 = vel(Wa + 0.5 * ha[:, None] * k2, ta + 0.5 * ha)
            k4 = vel(Wa + ha[:, None] * k3, ta + ha); Wp = Wa + ha[:, None] / 6 * (k1 + 2 * k2 + 2 * k3 + k4); tn = ta + ha
            conv = np.isfinite(Wp).all(1); d_prev = None
            for c in range(3):
                dW = _solve(HW(Wp, tn), Hfull(Wp, tn)[..., None])[..., 0]; Wp = Wp - dW
                nd = np.linalg.norm(dW, axis=1) / np.linalg.norm(Wp, axis=1)
                if d_prev is not None: conv &= (nd <= 0.5 * d_prev + 1e-14)
                d_prev = nd
            conv &= np.isfinite(Wp).all(1) & (d_prev < tol * 100)
        acc = A[conv]; rej = A[~conv]
        W[acc] = Wp[conv]; t[acc] = tn[conv]; steps[acc] += 1; streak[acc] += 1
        grow = acc[streak[acc] >= 5]; h[grow] = np.minimum(2 * h[grow], hmax); streak[grow] = 0
        h[rej] /= 2; streak[rej] = 0
        status[acc[t[acc] >= 1 - 1e-15]] = 0
        status[rej[h[rej] < hmin]] = 2
    status[status < 0] = 2
    track_projective.last_t = t
    return W, status, steps
