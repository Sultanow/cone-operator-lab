#!/bin/bash
#SBATCH --job-name=hankel-fullspec
#SBATCH --output=/home/esul01/data/outputs/logs/hankel_fullspec_%A_%a.out
#SBATCH --error=/home/esul01/data/outputs/logs/hankel_fullspec_%A_%a.err
#SBATCH --array=0-47%8
#SBATCH --requeue
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G

# ------------------------------------------------------------------
# Gap-free tile-parallel resolvent block-Hankel, full-spectrum run.
#
# Task layout: N_BANDS x N_SEEDS tasks, TASK = band * N_SEEDS + seed.
#   array=0-47  <=>  6 bands x 8 seeds.  If you change N_BANDS or the
#   seed list, adjust the #SBATCH --array line accordingly:
#   array upper bound = N_BANDS * N_SEEDS - 1.
#
# The lambda tiling (Weyl equal-count edges) is computed INSIDE
# run_hankel_band.py from (a, b, c, n-target, n-tiles); this script
# only selects (band, seed). This removes the bash/python duplication
# that previously caused window bounds to be computed but never passed.
# ------------------------------------------------------------------

set -euo pipefail

N_BANDS=6
SEEDS=(11 22 33 44 55 66 77 88)
N_SEEDS=${#SEEDS[@]}

TASK=${SLURM_ARRAY_TASK_ID}
BAND_INDEX=$((TASK / N_SEEDS))
SEED=${SEEDS[$((TASK % N_SEEDS))]}

OUT_DIR="/home/esul01/data/outputs/hankel_full_spectrum_triaxial"
mkdir -p /home/esul01/data/outputs/logs "${OUT_DIR}"

cd /home/esul01/hearing-ellipsoid-bench

source /home/jupyter/anaconda3/etc/profile.d/conda.sh
conda activate jupyterhub_env

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=1

echo "Job ${SLURM_JOB_ID}, task ${TASK}: band=${BAND_INDEX} seed=${SEED}"
echo "started=$(date)"
# Moment recursion checkpoints every 2000 steps into OUT_DIR;
# a resubmitted/requeued task resumes automatically.

python scripts/run_hankel_band.py \
  --data-root /home/esul01/data \
  --out-dir "${OUT_DIR}" \
  --a 1.0 \
  --b 1.5 \
  --c 2.3 \
  --mesh-h 0.06 \
  --order 2 \
  --n-target 2050 \
  --n-tiles 64 \
  --n-bands "${N_BANDS}" \
  --band-index "${BAND_INDEX}" \
  --seed "${SEED}" \
  --n-moments 0 \
  --resolution-factor 0.6 \
  --l-max 34000 \
  --n-probes 13 \
  --hankel-blocks 110 \
  --max-rank 500 \
  --rank-tol 1e-10 \
  --transition-frac 0.10 \
  --pad-frac 0.75

echo "finished=$(date)"

# After ALL array tasks are done, merge + audit on a login/short node:
#   python scripts/merge_hankel_full_spectrum.py \
#     --root /home/esul01/data/outputs/hankel_full_spectrum_triaxial \
#     --a 1.0 --b 1.5 --c 2.3 --min-seeds 5
# The audit writes tiles_to_rerun.txt for any tile with a count deficit.
