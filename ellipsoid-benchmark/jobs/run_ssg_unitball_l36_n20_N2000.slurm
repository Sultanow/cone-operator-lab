#!/bin/bash
#SBATCH --job-name=ssg-unitball-l36-n20
#SBATCH --output=/home/esul01/data/outputs/logs/ssg_unitball_l36_n20_q64_56_112_N2000_%j.out
#SBATCH --error=/home/esul01/data/outputs/logs/ssg_unitball_l36_n20_q64_56_112_N2000_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=700G

set -euo pipefail

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-unknown}"

mkdir -p /home/esul01/data/outputs/logs
mkdir -p /home/esul01/data/outputs/ssg_reference_sphere

cd /home/esul01/hearing-ellipsoid-bench

source /home/jupyter/anaconda3/etc/profile.d/conda.sh
conda activate jupyterhub_env

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export NUMEXPR_NUM_THREADS=1

python scripts/run_ssg_reference.py \
  --out-dir /home/esul01/data/outputs/ssg_reference_sphere \
  --a 1.0 \
  --b 1.0 \
  --c 1.0 \
  --l-max 36 \
  --n-max 20 \
  --n-radial 64 \
  --n-theta 56 \
  --n-phi 112 \
  --n-eigs 2000

echo "Job finished at $(date)"