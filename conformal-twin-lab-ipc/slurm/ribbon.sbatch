#!/bin/bash
#SBATCH --job-name=ct-ribbon
#SBATCH --array=1-64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=/home/esul01/data/outputs/logs/ct_ribbon_%A_%a.out
#SBATCH --error=/home/esul01/data/outputs/logs/ct_ribbon_%A_%a.err
#SBATCH --partition=std
# Ribbon-graph figure: 64 independent layout searches (different seeds); pick the best with scripts/pick_ribbon.py.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
TASK=${SLURM_ARRAY_TASK_ID:?run as an array job}
python3 "$CT_ROOT/visual/fig_ribbon.py" "$CT_OUT/ribbon/seed_$TASK" --seed "$TASK" --restarts 40 --iters 6000
echo "finished=$(date)"
