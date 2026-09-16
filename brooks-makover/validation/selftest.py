"""Fast independent validation checks.

This script deliberately checks properties of the validation modules rather
than importing production `bm_cusps` code.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hyperbolic_disk import fem_hyperbolic_disk, radial_ground_truth
from pants_atom import build_pants as pants_lengths, close_defect
from pants import build_pants
from genus_n import K4_genus3, build_surface


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"ok   {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also run the slower Bolza closed-surface benchmark")
    args = ap.parse_args()

    # 1. Hyperbolic disk: independent 1D/2D agreement at modest resolution.
    R = 1.0
    gt = radial_ground_truth(R, N=12000, k=2)[0]
    fem, _ = fem_hyperbolic_disk(R, nr=42, k=2)
    rel = abs(fem[0] - gt) / gt
    check(rel < 1.5e-2, f"hyperbolic disk FEM agrees with radial reference ({rel:.2%})")

    # 2. Right-angled hexagon closure over representative cuff lengths.
    sides, _ = pants_lengths(0.5, 2.0, 3.0)
    defect = close_defect(sides)
    check(defect < 1e-9, f"right-angled hexagon holonomy closes ({defect:.2e})")

    # 3. Pair of pants: area and zero mode.
    P = build_pants(1.0, 1.7, 2.4, m=16, h=0.05)
    area_rel = abs(P["area"] - 2 * math.pi) / (2 * math.pi)
    check(area_rel < 1.5e-2, f"pair-of-pants area is 2*pi within FEM tolerance ({area_rel:.2%})")
    check(abs(P["l0"]) < 1e-8, f"pair-of-pants Neumann zero mode is present ({P['l0']:.2e})")

    # 4. Genus 3 K4 gluing: closed topology sanity.
    L = np.array([2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
    T = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    edges, slots = K4_genus3(L, T)
    area, vals, _ = build_surface(edges, slots, m=9, h=0.085, k=4)
    target = 8 * math.pi
    area_rel = abs(area - target) / target
    check(area_rel < 2.0e-2, f"genus-3 area is 8*pi within FEM tolerance ({area_rel:.2%})")
    check(abs(vals[0]) < 1e-8, f"closed genus-3 zero mode is present ({vals[0]:.2e})")
    check(vals[1] > 0, f"closed genus-3 first non-zero eigenvalue is positive ({vals[1]:.4f})")

    if args.full:
        proc = subprocess.run(
            [sys.executable, str(HERE / "bolza_closed.py")],
            text=True, capture_output=True
        )
        check(proc.returncode == 0, "Bolza benchmark executes successfully")
        # Keep this as an external-process check because bolza_closed.py is intentionally
        # a standalone independent benchmark.
        for line in proc.stdout.splitlines():
            if "lambda_1 =" in line and "Bolza reference" in line:
                print("     " + line.strip())
                break

    print("\nALL VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
