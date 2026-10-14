#!/bin/bash
#SBATCH --job-name=ct-ipc
#SBATCH --array=1-16
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=/home/esul01/data/outputs/logs/ct_ipc_%A_%a.out
#SBATCH --error=/home/esul01/data/outputs/logs/ct_ipc_%A_%a.err
#SBATCH --partition=std
# IPC sweep: contact-aware solver, levels 1 -> 2 -> 3 -> 4, 16 tasks.
# Checkpoints after every level; resubmitting the same array resumes finished levels.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
GRID=${CT_GRID:-$CT_ROOT/grid_ipc.txt}
CT_RUN=sweep_ipc
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
