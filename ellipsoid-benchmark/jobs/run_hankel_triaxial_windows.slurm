#!/bin/bash
#SBATCH --job-name=hankel-window
#SBATCH --output=/home/esul01/data/outputs/logs/hankel_window_%A_%a.out
#SBATCH --error=/home/esul01/data/outputs/logs/hankel_window_%A_%a.err
#SBATCH --array=0-7%4
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G

set -euo pipefail

mkdir -p /home/esul01/data/outputs/logs
cd /home/esul01/hearing-ellipsoid-bench

source /home/jupyter/anaconda3/etc/profile.d/conda.sh
conda activate jupyterhub_env

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=1

# 30 overlapping lambda windows times 8 probe seeds.
# lambda window width = 25, step = 15, range approximately [0, 450].
TASK=${SLURM_ARRAY_TASK_ID}
WINDOW_ID=$((TASK / 8))
SEED_ID=$((TASK % 8))

SEEDS=(11 22 33 44 55 66 77 88)
SEED=${SEEDS[$SEED_ID]}

read LAMBDA_BOTTOM LAMBDA_TOP <<< "$(python - <<PY
window_id = ${WINDOW_ID}
global_min = 0.0
global_max = 450.0
step = 15.0
width = 25.0
lo = global_min + window_id * step
hi = min(lo + width, global_max)
print(f"{lo:.10f} {hi:.10f}")
PY
)"

OUT_DIR="/home/esul01/data/outputs/hankel_resolvent_triaxial_windows/window_$(printf '%02d' ${WINDOW_ID})/seed_${SEED}"
mkdir -p "${OUT_DIR}"

echo "Job ${SLURM_JOB_ID}, task ${TASK}"
echo "window_id=${WINDOW_ID}"
echo "seed=${SEED}"
echo "lambda interval=[${LAMBDA_BOTTOM}, ${LAMBDA_TOP}]"
echo "started=$(date)"

python scripts/run_hankel_seed.py \
  --data-root /home/esul01/data \
  --out-dir "${OUT_DIR}" \
  --a 1.0 \
  --b 1.5 \
  --c 2.3 \
  --mesh-h 0.06 \
  --order 2 \
  --lambda-bottom 0 \
  --lambda-top 450 \
  --lambda-max 450 \
  --sigma 100 \
  --n-moments 3000 \
  --n-windows 36 \
  --n-probes 13 \
  --hankel-blocks 110 \
  --max-rank 450 \
  --rank-tol 1e-10 \
  --transition-frac 0.10 \
  --window-overlap 0.35 \
  --seed "${SEED}"

echo "finished=$(date)"
