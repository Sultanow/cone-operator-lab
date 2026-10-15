#!/bin/bash
#SBATCH --job-name=ct-groebner
#SBATCH --array=1-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/home/esul01/data/outputs/logs/ct_groebner_%A_%a.out
#SBATCH --error=/home/esul01/data/outputs/logs/ct_groebner_%A_%a.err
#SBATCH --partition=std
# Deterministic Groebner basis over Q of the C4-symmetric realisation system (Singular, exact rational arithmetic).
# Task 1: std, task 2: slimgb. Input from: python src/symm/msolve_input.py $CT_OUT/algebraic
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
ALG=$([ "${SLURM_ARRAY_TASK_ID:-1}" = 1 ] && echo std || echo slimgb)
SINGULAR=${CT_SINGULAR:-Singular}
"$SINGULAR" -q "$CT_OUT/algebraic/symm_$ALG.sing" > "$CT_OUT/algebraic/singular_$ALG.log"
echo "finished=$(date)"
