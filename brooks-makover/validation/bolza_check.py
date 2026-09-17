"""Hard-check wrapper for the independent Bolza benchmark.

The underlying ``bolza_closed.py`` remains a standalone implementation.  This
wrapper executes it and turns its printed diagnostics into explicit assertions,
so a degraded geometry or spectrum cannot silently pass CI.
"""
from __future__ import annotations
import re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOLZA_REF = 3.838887258


def run_and_check(max_rel_lambda1=0.005, max_pairing_err=1e-10,
                  max_isometry_defect=1e-10, max_zero_mode=1e-7):
    proc = subprocess.run([sys.executable, str(HERE / 'bolza_closed.py')],
                          text=True, capture_output=True)
    if proc.returncode != 0:
        raise AssertionError('bolza_closed.py failed:\n' + proc.stderr)
    out = proc.stdout

    iso = [abs(float(x)) for x in re.findall(r"\|p\|-\|q\|=([+-]?[0-9.eE+-]+)", out)]
    ep = [float(x) for x in re.findall(r"endpoint err=([0-9.eE+-]+)", out)]
    mcls = re.search(r"vertex classes:\s*(\d+)", out)
    mz = re.search(r"lambda_0\s*=\s*([+-]?[0-9.eE+-]+)", out)
    m1 = re.search(r"lambda_1\s*=\s*([0-9.eE+-]+)", out)
    if not (iso and ep and mcls and mz and m1):
        raise AssertionError('could not parse all Bolza diagnostics')

    classes = int(mcls.group(1)); lam0 = float(mz.group(1)); lam1 = float(m1.group(1))
    rel = abs(lam1 - BOLZA_REF) / BOLZA_REF
    if max(iso) > max_isometry_defect:
        raise AssertionError(f'Bolza side-pairing isometry defect {max(iso):.3e}')
    if max(ep) > max_pairing_err:
        raise AssertionError(f'Bolza endpoint pairing error {max(ep):.3e}')
    if classes != 1:
        raise AssertionError(f'Bolza vertex classes={classes}, expected 1')
    if abs(lam0) > max_zero_mode:
        raise AssertionError(f'Bolza zero mode {lam0:.3e} exceeds tolerance')
    if rel > max_rel_lambda1:
        raise AssertionError(f'Bolza lambda_1 relative error {rel:.3%} exceeds {max_rel_lambda1:.3%}')
    return dict(lambda0=lam0, lambda1=lam1, relative_error=rel,
                max_isometry_defect=max(iso), max_pairing_error=max(ep),
                vertex_classes=classes, stdout=out)
