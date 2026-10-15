"""Homotopy continuation for ALL complex solutions of the C4-symmetric realisation problem.
The 22 orbit equations are randomised to 17 (fixed complex matrix A, seed 2026); a run is defined by
the gamma seed; the 2^17 start points are split into --nbatches batches. Output: finite end points."""
import argparse, json, os, sys, time, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from symsys import residuals, jacobian, DV
from track import track_projective, newton
ap = argparse.ArgumentParser()
ap.add_argument('--run', type=int, default=0); ap.add_argument('--batch', type=int, default=0)
ap.add_argument('--nbatches', type=int, default=64); ap.add_argument('--out', type=str, default='out_symm')
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
n = 17; rngA = np.random.default_rng(2026)
A = (rngA.normal(size=(n, 22)) + 1j * rngA.normal(size=(n, 22))) / np.sqrt(44)
F = lambda z: residuals(z) @ A.T
JF = lambda z: np.einsum('ij,bjk->bik', A, jacobian(z))
rg = np.random.default_rng(1000 + args.run)
gamma = np.exp(2j * np.pi * rg.random()); v = rg.normal(size=n + 1) + 1j * rg.normal(size=n + 1)
def Fh(W):   # the equations are |x_k - T^d x_l|^2 - D: homogeneous quadratic part minus a constant
    w0, z = W[:, [0]], W[:, 1:]; return (residuals(z) + DV - DV * w0 ** 2) @ A.T
def JFh(W):
    w0, z = W[:, [0]], W[:, 1:]; Jz = np.einsum('ij,bjk->bik', A, jacobian(z)); J0 = ((-2 * DV) * w0) @ A.T
    return np.concatenate([J0[:, :, None], Jz], 2)
allstarts = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
idx = np.array_split(np.arange(len(allstarts)), args.nbatches)[args.batch]
t0 = time.time(); Wend, st, steps = track_projective(Fh, JFh, n, allstarts[idx], gamma, v)
w0 = Wend[:, 0]; tt = track_projective.last_t; rel0 = np.abs(w0) / np.linalg.norm(Wend, axis=1)
fin = np.where((st == 0) & (rel0 > 1e-7))[0]; atinf = int(np.sum((st == 0) & (rel0 <= 1e-7)))
fail_inf = (st == 2) & (tt > 0.999) & (rel0 < 1e-2)        # singular end points at infinity: irrelevant
fail_end = (st == 2) & (tt > 0.999) & (rel0 >= 1e-2)       # singular FINITE end points (would include flexible realisations)
fail_mid = (st == 2) & (tt <= 0.999)                       # genuine tracking failures: covered by the other runs
zf = newton(F, JF, Wend[fin][:, 1:] / Wend[fin][:, [0]]) if len(fin) else np.zeros((0, n), complex)
resF = np.abs(F(zf)).max(1); resO = np.abs(residuals(zf)).max(1)
cond = np.array([np.linalg.cond(J) for J in JF(zf)]) if len(zf) else np.array([])
out = dict(run=args.run, batch=args.batch, paths=int(len(idx)), finite=int(len(fin)), at_infinity=atinf + int(fail_inf.sum()), singular_finite=int(fail_end.sum()), failed=int(fail_mid.sum()),
           seconds=round(time.time() - t0, 1), median_steps=int(np.median(steps)),
           z_re=zf.real.tolist(), z_im=zf.imag.tolist(), res_randomised=resF.tolist(), res_original=resO.tolist(), cond=cond.tolist(),
           failed_start_indices=idx[fail_mid].tolist(), singular_finite_re=(Wend[fail_end][:, 1:] / Wend[fail_end][:, [0]]).real.tolist(), singular_finite_im=(Wend[fail_end][:, 1:] / Wend[fail_end][:, [0]]).imag.tolist())
json.dump(out, open(f'{args.out}/symm_run{args.run}_batch{args.batch:03d}.json', 'w'))
print(json.dumps({k: out[k] for k in ('run', 'batch', 'paths', 'finite', 'at_infinity', 'singular_finite', 'failed', 'seconds', 'median_steps')}))
