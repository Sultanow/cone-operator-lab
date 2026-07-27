#!/bin/bash
#SBATCH --job-name=arnoldi-unitball-h006
#SBATCH --output=/home/esul01/data/outputs/logs/arnoldi_unitball_h006_N2000_%j.out
#SBATCH --error=/home/esul01/data/outputs/logs/arnoldi_unitball_h006_N2000_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G

set -euo pipefail

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-unknown}"

mkdir -p /home/esul01/data/outputs/logs
mkdir -p /home/esul01/data/outputs/arnoldi_reference_sphere

cd /home/esul01/hearing-ellipsoid-bench

source /home/jupyter/anaconda3/etc/profile.d/conda.sh
conda activate jupyterhub_env

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/run_arnoldi_reference.py \
  --data-root /home/esul01/data \
  --out-dir /home/esul01/data/outputs/arnoldi_reference_sphere \
  --a 1.0 \
  --b 1.0 \
  --c 1.0 \
  --mesh-h 0.06 \
  --order 2 \
  --n-eigs 2000 \
  --tol 1e-10 \
  --ncv-factor 2.0 \
  --min-ncv 120

echo "Job finished at $(date)"
