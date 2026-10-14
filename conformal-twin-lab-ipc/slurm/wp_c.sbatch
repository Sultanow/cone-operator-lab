#!/bin/bash
#SBATCH --job-name=ct-wpc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/esul01/data/outputs/logs/ct_wpc_%j.out
#SBATCH --error=/home/esul01/data/outputs/logs/ct_wpc_%j.err
#SBATCH --partition=std
# Work package C: h_P (polyhedron's own conformal structure) versus h_cp (circle packing).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
python3 "$CT_ROOT/src/wpc/wp_c.py" "$CT_OUT/wp_c"
echo "finished=$(date)"
