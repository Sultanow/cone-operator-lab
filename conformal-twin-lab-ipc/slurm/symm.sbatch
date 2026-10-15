#!/bin/bash
#SBATCH --job-name=ct-symm
#SBATCH --array=1-384%96
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=/home/esul01/data/outputs/logs/ct_symm_%A_%a.out
#SBATCH --error=/home/esul01/data/outputs/logs/ct_symm_%A_%a.err
#SBATCH --partition=std
# Complete C4-symmetric classification: homotopy continuation over all 2^17 start points,
# three independent runs (different gamma), 128 batches each. Task t: run (t-1)/128, batch (t-1)%128.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
TASK=${SLURM_ARRAY_TASK_ID:?run as an array job}
RUN=$(( (TASK - 1) / 128 )); BATCH=$(( (TASK - 1) % 128 ))
OUTDIR="$CT_OUT/symmetric"
mkdir -p "$OUTDIR"
echo "task=$TASK run=$RUN batch=$BATCH"
python3 "$CT_ROOT/src/symm/run_symm.py" --run "$RUN" --batch "$BATCH" --nbatches 128 --out "$OUTDIR"
echo "finished=$(date)"
