#!/bin/bash
#SBATCH --job-name=hankel-h006-scout
#SBATCH --output=/home/esul01/data/outputs/logs/hankel_h006_scout_%A_%a.out
#SBATCH --error=/home/esul01/data/outputs/logs/hankel_h006_scout_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --array=0-7%2

set -euo pipefail

SEEDS=(11 22 33 44 55 66 77 88)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Seed=${SEED}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-unknown}"

mkdir -p /home/esul01/data/outputs/logs
mkdir -p /home/esul01/data/outputs/hankel_reference_triaxial

cd /home/esul01/hearing-ellipsoid-bench

source /home/jupyter/anaconda3/etc/profile.d/conda.sh
conda activate jupyterhub_env

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Sparse LU and repeated solves can behave badly if every array task also spawns 64 BLAS threads.
# Keep BLAS conservative; parallelism comes from the Slurm array.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/run_hankel_seed.py \
  --data-root /home/esul01/data \
  --out-dir /home/esul01/data/outputs/hankel_reference_triaxial \
  --a 1.0 \
  --b 1.5 \
  --c 2.3 \
  --mesh-h 0.06 \
  --order 2 \
  --lambda-top 5500 \
  --n-moments 2500 \
  --n-windows 10 \
  --n-probes 9 \
  --hankel-blocks 90 \
  --max-rank 350 \
  --rank-tol 1e-9 \
  --transition-frac 0.12 \
  --window-overlap 0.20 \
  --seed ${SEED}

echo "Job finished at $(date)"