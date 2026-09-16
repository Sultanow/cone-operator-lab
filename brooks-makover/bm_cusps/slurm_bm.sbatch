#!/bin/bash
#SBATCH --job-name=bm-cusps
#SBATCH --array=1-80%16
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/bm_%A_%a.out
#SBATCH --error=logs/bm_%A_%a.err
#
# One task = one (n, seed, h) sample of the Brooks-Makover model:
#   cusped surface S (exact cusp DtN)  vs.  uniformized compactification S^bar.
# Submit:   python make_params.py --n 16 32 64 128 --seeds 20 --h 0.1 0.07
#           mkdir -p logs results; sbatch --array=1-$(wc -l < params.txt)%16 slurm_bm.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- environment (adapt to your cluster) ---------------------------------------
module load python/3.11 2>/dev/null || true
if [ -d .venv ]; then source .venv/bin/activate; fi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" params.txt)
read -r n seed h L0 <<< "$line"
out="results/bm_n${n}_s${seed}_h${h}.json"
mkdir -p results
if [ -s "$out" ]; then
  if python validate_result.py "$out" --n "$n" --seed "$seed" --h "$h"; then
    echo "validated existing result; skip: $out"
    exit 0
  fi
  bad="${out%.json}.invalid.$(date +%Y%m%dT%H%M%S).json"
  echo "existing result is stale/incomplete/mismatched; moving to $bad and recomputing" >&2
  mv "$out" "$bad"
fi

echo "task $SLURM_ARRAY_TASK_ID: n=$n seed=$seed h=$h L0=$L0  ($(hostname), $(date))"
python run_bm.py --n "$n" --seed "$seed" --h "$h" --L0 "$L0" --neig 60 --out "$out"
