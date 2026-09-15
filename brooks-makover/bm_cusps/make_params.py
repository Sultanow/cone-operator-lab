#!/usr/bin/env python3
"""Write params.txt for the SLURM job array: one line  'n seed h L0'  per task."""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, nargs="+", default=[16, 32, 64, 128, 256])
ap.add_argument("--seeds", type=int, default=16)
ap.add_argument("--h", type=float, nargs="+", default=[0.1], help="one or two mesh sizes (two -> Richardson)")
ap.add_argument("--L0", type=float, default=1.0)
ap.add_argument("--out", default="params.txt")
a = ap.parse_args()
lines = ["%d %d %g %g" % (n, s, h, a.L0) for n in a.n for s in range(1, a.seeds + 1) for h in a.h]
open(a.out, "w").write("\n".join(lines) + "\n")
print("%d tasks -> %s   (sbatch --array=1-%d%%16 slurm_bm.sbatch)" % (len(lines), a.out, len(lines)))
