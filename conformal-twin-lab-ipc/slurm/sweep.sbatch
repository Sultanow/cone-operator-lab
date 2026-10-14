#!/bin/bash
#SBATCH --job-name=ct-sweep
#SBATCH --array=1-96%32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/esul01/data/outputs/logs/ct_sweep_%A_%a.out
#SBATCH --error=/home/esul01/data/outputs/logs/ct_sweep_%A_%a.err
##SBATCH --partition=CHANGE_ME
# One task = one line of the grid file = one multilevel run (levels 2 -> 3 -> 4).
# Checkpoints after every level; resubmitting the same array resumes finished levels.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
GRID=${CT_GRID:-$CT_ROOT/grid.txt}
TASK=${SLURM_ARRAY_TASK_ID:?run as an array job}
NLINES=$(wc -l < "$GRID")
if [ "$TASK" -gt "$NLINES" ]; then
  echo "task $TASK exceeds the $NLINES lines of $GRID"; exit 1
fi
read -r -a ARGV <<< "$(sed -n "${TASK}p" "$GRID")"
OUTDIR="$CT_OUT/$CT_RUN"
mkdir -p "$OUTDIR"
echo "task=$TASK args=${ARGV[*]}"
python3 "$CT_ROOT/src/conformal_embed.py" "${ARGV[@]}" --out "$OUTDIR" --resume
echo "finished=$(date)"
