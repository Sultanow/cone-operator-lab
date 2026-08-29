#!/bin/bash
#SBATCH --job-name=ballrank-test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=36:00:00
#SBATCH --output=/home/esul01/data/outputs/logs/%x_%j.out

source /home/jupyter/anaconda3/etc/profile.d/conda.sh
conda activate jupyterhub_env

cd /home/esul01/hearing-ellipsoid-bench
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python scripts/run_hankel_band.py --data-root /home/esul01/data \
  --out-dir /home/esul01/data/outputs/ball_rank_test \
  --a 1.0 --b 1.0 --c 1.0 --mesh-h 0.06 --order 2 \
  --n-target 2050 --n-tiles 64 --n-bands 6 --band-index 4 --seed 11 \
  --n-probes 64 --hankel-blocks 110 --max-rank 500
