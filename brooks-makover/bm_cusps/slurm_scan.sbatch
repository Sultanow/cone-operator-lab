#!/bin/bash
#SBATCH --job-name=bm-scan
#SBATCH --array=0-99
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/scan_%A_%a.out
# Stage 1 of the funnel: graph features only, 10^4 seeds per task (~0.07 s/graph at n=64).
# Usage:  N=128 sbatch slurm_scan.sbatch ; afterwards  cat scan/scan_n128_*.csv > scan_n128.csv
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11 2>/dev/null || true
if [ -d .venv ]; then source .venv/bin/activate; fi
export OMP_NUM_THREADS=1
N=${N:-128}; PER=${PER:-10000}
a=$((SLURM_ARRAY_TASK_ID * PER)); b=$((a + PER))
mkdir -p scan
python graph_scan.py --n "$N" --seeds "$a" "$b" --W 10 --out "scan/scan_n${N}_${SLURM_ARRAY_TASK_ID}.csv"
