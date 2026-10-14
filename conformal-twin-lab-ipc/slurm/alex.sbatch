#!/bin/bash
#SBATCH --job-name=ct-alex
#SBATCH --array=1-128%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=/home/esul01/data/outputs/logs/ct_alex_%A_%a.out
#SBATCH --error=/home/esul01/data/outputs/logs/ct_alex_%A_%a.err
#SBATCH --partition=std
# Alexandrov-type search: are there further polyhedral realisations of the intrinsic metric of P?
# Odd tasks: triangulation K. Even tasks: nonagons folded along a random other set of diagonals (one per task).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
TASK=${SLURM_ARRAY_TASK_ID:?run as an array job}
N=${CT_ALEX_N:-50000}
OUTDIR="$CT_OUT/alexandrov"
mkdir -p "$OUTDIR"
EXTRA=""
if [ $((TASK % 2)) -eq 0 ]; then EXTRA="--retriangulate"; fi
echo "task=$TASK n=$N extra=$EXTRA"
python3 "$CT_ROOT/src/alex/search.py" --seed "$TASK" --n "$N" $EXTRA --out "$OUTDIR"
echo "finished=$(date)"
