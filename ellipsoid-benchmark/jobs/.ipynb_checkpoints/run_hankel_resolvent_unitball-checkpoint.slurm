#!/bin/bash
#SBATCH --job-name=hankel-res-unitball
#SBATCH --output=/home/esul01/data/outputs/logs/hankel_res_unitball_%A_%a.out
#SBATCH --error=/home/esul01/data/outputs/logs/hankel_res_unitball_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --array=0-7%4

set -euo pipefail

SEEDS=(11 22 33 44 55 66 77 88)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Seed=${SEED}"
echo "CPUs allocated=${SLURM_CPUS_PER_TASK:-unknown}"

mkdir -p /home/esul01/data/outputs/logs
mkdir -p /home/esul01/data/outputs/hankel_resolvent_sphere

cd /home/esul01/hearing-ellipsoid-bench

source /home/jupyter/anaconda3/etc/profile.d/conda.sh
conda activate jupyterhub_env

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Sparse SuperLU dominates the moment iteration.
# Seed-level parallelism supplies the main speedup.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=1

python scripts/run_hankel_seed.py \
  --data-root /home/esul01/data \
  --out-dir /home/esul01/data/outputs/hankel_resolvent_sphere \
  --a 1.0 \
  --b 1.0 \
  --c 1.0 \
  --mesh-h 0.06 \
  --order 2 \
  --lambda-bottom 0 \
  --lambda-top 1020 \
  --sigma 100 \
  --n-moments 3000 \
  --n-windows 12 \
  --n-probes 13 \
  --hankel-blocks 110 \
  --max-rank 450 \
  --rank-tol 1e-10 \
  --transition-frac 0.10 \
  --window-overlap 0.25 \
  --seed ${SEED}

echo "Job finished at $(date)"