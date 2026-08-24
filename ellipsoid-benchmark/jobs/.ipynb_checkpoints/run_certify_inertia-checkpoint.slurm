#!/bin/bash
#SBATCH --job-name=certify-inertia
#SBATCH --array=0-15%16
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=36:00:00
#SBATCH --output=%x_%A_%a.out

# Per-cluster inertia certification of the merged Hankel spectrum.
# ~2035 probes split over 16 tasks (~128 each); each probe is one
# MUMPS LDL^T numeric factorization (analysis reused). Idempotent:
# resubmission skips completed chunks.
#
# Afterwards, ONCE on a single node / interactively:
#   python scripts/certify_inertia.py --mode assemble \
#     --root  /home/esul01/data/outputs/hankel_full_spectrum_triaxial \
#     --data-root /home/esul01/data \
#     --a 1.0 --b 1.5 --c 2.3 --mesh-h 0.06 --order 2

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/esul01/hearing-ellipsoid-bench/src:${PYTHONPATH}

cd /home/esul01/hearing-ellipsoid-bench
echo "task=${SLURM_ARRAY_TASK_ID} started=$(date)"

python scripts/certify_inertia.py --mode probes \
  --root  /home/esul01/data/outputs/hankel_full_spectrum_triaxial \
  --data-root /home/esul01/data \
  --a 1.0 --b 1.5 --c 2.3 --mesh-h 0.06 --order 2 \
  --task-index ${SLURM_ARRAY_TASK_ID} --n-tasks 16

echo "finished=$(date)"
