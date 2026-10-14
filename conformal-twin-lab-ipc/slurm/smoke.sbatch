#!/bin/bash
#SBATCH --job-name=ct-smoke
#SBATCH --array=1-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/esul01/data/outputs/logs/ct_smoke_%A_%a.out
#SBATCH --error=/home/esul01/data/outputs/logs/ct_smoke_%A_%a.err
##SBATCH --partition=CHANGE_ME
# Two short runs (levels 1 -> 2, 40 iterations per stage) to validate the environment.
set -euo pipefail
export CT_GRID=${CT_GRID:-${SLURM_SUBMIT_DIR:-$PWD}/grid_smoke.txt}
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
CT_RUN=smoke
GRID=$CT_GRID
TASK=${SLURM_ARRAY_TASK_ID:?run as an array job}
read -r -a ARGV <<< "$(sed -n "${TASK}p" "$GRID")"
OUTDIR="$CT_OUT/$CT_RUN"
mkdir -p "$OUTDIR"
echo "task=$TASK args=${ARGV[*]}"
python3 "$CT_ROOT/src/conformal_embed.py" "${ARGV[@]}" --out "$OUTDIR"
echo "finished=$(date)"
