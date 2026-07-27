#!/bin/bash
#SBATCH --job-name=mfs-triaxial
#SBATCH --output=/home/esul01/data/outputs/logs/mfs_triaxial_%j.out
#SBATCH --error=/home/esul01/data/outputs/logs/mfs_triaxial_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G

set -euo pipefail

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "CPUs allocated=${SLURM_CPUS_PER_TASK:-unknown}"

mkdir -p /home/esul01/data/outputs/logs
mkdir -p /home/esul01/data/outputs/boundary_mfs_triaxial

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
  --out-dir /home/esul01/data/outputs/boundary_mfs_triaxial \
  --a 1.0 \
  --b 1.5 \
  --c 2.3 \
  --k-min 2.0 \
  --k-max 22.0 \
  --n-scan 16000 \
  --n-boundary 2400 \
  --n-sources 1800 \
  --n-interior 900 \
  --source-scale 1.30 \
  --tension-threshold 0.004 \
  --relative-gap 5e-5 \
  --refinement-xatol 1e-11

echo "Job finished at $(date)"
