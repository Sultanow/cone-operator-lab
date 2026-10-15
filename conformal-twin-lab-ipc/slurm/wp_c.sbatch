#!/bin/bash
#SBATCH --job-name=ct-wpc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/esul01/data/outputs/logs/ct_wpc_%j.out
#SBATCH --error=/home/esul01/data/outputs/logs/ct_wpc_%j.err
#SBATCH --partition=std
# h_P (the polyhedron's own conformal structure) versus h_cp, and the exact way back (Troyanov).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
python3 "$CT_ROOT/src/wpc/wp_c.py" "$CT_OUT/wp_c"          # Proposition 6: h_P against h_cp
python3 "$CT_ROOT/src/wpc/roundtrip.py" "$CT_OUT/wp_c"     # Table 3: the exact way back
echo "finished=$(date)"
