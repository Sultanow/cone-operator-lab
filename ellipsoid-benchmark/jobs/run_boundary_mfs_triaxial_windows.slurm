#!/bin/bash
#SBATCH --job-name=mfs-window
#SBATCH --output=/home/esul01/data/outputs/logs/mfs_window_%A_%a.out
#SBATCH --error=/home/esul01/data/outputs/logs/mfs_window_%A_%a.err
#SBATCH --array=0-79%8
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

set -euo pipefail

mkdir -p /home/esul01/data/outputs/logs
cd /home/esul01/hearing-ellipsoid-bench

source /home/jupyter/anaconda3/etc/profile.d/conda.sh
conda activate jupyterhub_env

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export NUMEXPR_NUM_THREADS=1

TASK=${SLURM_ARRAY_TASK_ID}
WINDOW_ID=$((TASK % 40))
SCALE_ID=$((TASK / 40))

read K_MIN K_MAX SOURCE_SCALE <<< "$(python - <<PY
window_id = ${WINDOW_ID}
scale_id = ${SCALE_ID}
global_min = 2.0
global_max = 22.0
step = 0.5
width = 0.6
scales = (1.25, 1.35)
k_min = global_min + window_id * step
k_max = min(k_min + width, global_max)
print(f"{k_min:.10f} {k_max:.10f} {scales[scale_id]:.2f}")
PY
)"

OUT_DIR="/home/esul01/data/outputs/boundary_mfs_triaxial_windows/scale_${SOURCE_SCALE}/window_$(printf '%02d' ${WINDOW_ID})"
mkdir -p "${OUT_DIR}"

echo "Job ${SLURM_JOB_ID}, task ${TASK}"
echo "window_id=${WINDOW_ID}"
echo "source_scale=${SOURCE_SCALE}"
echo "k interval=[${K_MIN}, ${K_MAX}]"
echo "started=$(date)"

python scripts/run_boundary_mfs.py \
  --out-dir "${OUT_DIR}" \
  --a 1.0 \
  --b 1.5 \
  --c 2.3 \
  --k-min "${K_MIN}" \
  --k-max "${K_MAX}" \
  --n-scan 2400 \
  --n-boundary 2400 \
  --n-sources 1800 \
  --n-interior 900 \
  --source-scale "${SOURCE_SCALE}" \
  --tension-threshold 1.0 \
  --relative-gap 5e-5 \
  --refinement-xatol 1e-11

echo "finished=$(date)"
