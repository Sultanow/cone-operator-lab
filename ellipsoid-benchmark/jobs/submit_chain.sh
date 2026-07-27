#!/bin/bash
# Submit the full-spectrum array N times as a dependency chain.
# Each pass resumes every unfinished (band, seed) task from its moment
# checkpoint; finished tasks exit immediately (candidates CSV present).
# 5 passes x 36 h covers band 5 (L = 34000 at ~16 s/step ~ 151 h).
set -euo pipefail
N_PASSES=${1:-5}
SCRIPT="$(dirname "$0")/run_hankel_full_spectrum_array.slurm"
DEP=""
for i in $(seq 1 "${N_PASSES}"); do
  if [ -z "${DEP}" ]; then
    JID=$(sbatch --parsable "${SCRIPT}")
  else
    JID=$(sbatch --parsable --dependency=afterany:"${DEP}" "${SCRIPT}")
  fi
  echo "pass ${i}: job ${JID}"
  DEP="${JID}"
done
echo "After the last pass finishes, run the merge (see script footer)."
