"""Single source of truth for code and result-schema versions.

SCHEMA_VERSION is written into every run_bm.py JSON; analyze.py and aggregate.py refuse
older schemas with a clear message instead of failing on a missing key.  Bump
SCHEMA_VERSION whenever a JSON key is renamed or its meaning changes."""
__version__ = "0.3.0"
SCHEMA_VERSION = 3
