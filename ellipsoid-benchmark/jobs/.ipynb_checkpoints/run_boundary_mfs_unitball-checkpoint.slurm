#!/bin/bash
#SBATCH --job-name=mfs-unitball
#SBATCH --output=/home/esul01/data/outputs/logs/mfs_unitball_%j.out
#SBATCH --error=/home/esul01/data/outputs/logs/mfs_unitball_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "CPUs allocated=${SLURM_CPUS_PER_TASK:-unknown}"

mkdir -p /home/esul01/data/outputs/logs
mkdir -p /home/esul01/data/outputs/boundary_mfs_sphere

cd /home/esul01/hearing-ellipsoid-bench

source /home/jupyter/anaconda3/etc/profile.d/conda.sh
conda activate jupyterhub_env

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Dense QR/SVD operations are BLAS/LAPACK dominated.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=1

python scripts/run_boundary_mfs.py \
  --out-dir /home/esul01/data/outputs/boundary_mfs_sphere \
  --a 1.0 \
  --b 1.0 \
  --c 1.0 \
  --k-min 2.5 \
  --k-max 15.0 \
  --n-scan 500 \
  --n-boundary 420 \
  --n-sources 300 \
  --n-interior 180 \
  --source-scale 1.30 \
  --tension-threshold 2e-3 \
  --relative-gap 2e-4 \
  --refinement-xatol 1e-10

echo "Job finished at $(date)"
