#!/bin/bash
#SBATCH --job-name=ct-certify
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/home/esul01/data/outputs/logs/ct_certify_%j.out
#SBATCH --error=/home/esul01/data/outputs/logs/ct_certify_%j.err
##SBATCH --partition=CHANGE_ME
# Computer-assisted proof of the circle-packing radii (Krawczyk test in Arb).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
mkdir -p "$CT_OUT/certificate"
for prec in 256 512; do
  python3 "$CT_ROOT/src/certify_radii.py" "$prec" "$CT_OUT/certificate/certified_radii_${prec}bit.json"
done
echo "finished=$(date)"
