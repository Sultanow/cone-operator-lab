# Result provenance policy

Two kinds of JSON files may exist in this repository and must not be conflated.

## Native production results

A result produced by the current `run_bm.py` carries

```json
"provenance": {
  "kind": "native_run",
  "numerical_payload_recomputed": true,
  "numerical_payload_code_version": "0.4.3",
  "metadata_migrated": false
}
```

Only such files are eligible for SLURM resume/reuse by `validate_result.py`.

## Archived example results

The 15 files under `results_example/` are historical smoke-test/example payloads. Their
numerical quantities (eigenvalues, profiles, mesh counts, timings) were **not recomputed**
when result schemas and quality metadata were updated. They therefore carry
`kind = archived_example_metadata_migration` and `numerical_payload_recomputed = false`.

They may be used to exercise plotting/aggregation code, but must not be described in a
paper as results computed with v0.4.3 (or earlier metadata-only migration versions). For
scientific tables/figures, use native production results and record the exact code commit,
parameters and result provenance.
