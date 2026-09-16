#!/usr/bin/env python3
"""Repository-root wrapper for the canonical bm_cusps self-test.

Run from either the repository root or bm_cusps/.  The actual test suite lives in
bm_cusps/selftest.py so there is exactly one implementation to maintain.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST = HERE / "bm_cusps" / "selftest.py"
if not TEST.exists():
    raise SystemExit("canonical self-test not found: %s" % TEST)
raise SystemExit(subprocess.call([sys.executable, str(TEST)] + sys.argv[1:], cwd=str(TEST.parent)))
