"""Single source of truth for code, result, scan, and graph-feature versions.

SCHEMA_VERSION is written into every run_bm.py JSON; analyze.py and aggregate.py refuse
older result schemas. SCAN_SCHEMA_VERSION and GRAPH_FEATURE_VERSION protect the stage-1
-> stage-2 boundary: a scan generated with older combinatorial semantics must never be
silently relabelled as a current result.
"""
__version__ = "0.4.5"
SCHEMA_VERSION = 6
SCAN_SCHEMA_VERSION = 1
# Bump whenever graph_features() changes meaning, even if its keys stay the same.
# Version 2 includes the repaired primitive-geodesic counting/completeness cutoff.
GRAPH_FEATURE_VERSION = 2
