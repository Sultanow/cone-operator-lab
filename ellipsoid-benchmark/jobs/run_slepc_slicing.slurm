#!/bin/bash
#SBATCH --job-name=slepc-slicing
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=360G
#SBATCH --time=36:00:00
#SBATCH --output=%x_%j.out

# Inertia-based spectrum slicing (SLEPc + MUMPS) on the SAME matrices
# as the FEM-ARPACK, block-Hankel, and certification runs.
#
# 64 MPI ranks are split into 8 sub-communicators (--partitions 8), so
# eight slices are processed concurrently with 8 ranks each.
#
# Environment (adapt to the cluster):
#   module load petsc slepc mumps    # or: conda activate slepc-env

source /home/jupyter/anaconda3/etc/profile.d/conda.sh
conda activate slepc-env

export OMP_NUM_THREADS=1
export PYTHONPATH=/home/esul01/hearing-ellipsoid-bench/src:${PYTHONPATH}

cd /home/esul01/hearing-ellipsoid-bench
echo "started=$(date)"

mpirun -n ${SLURM_NTASKS} python scripts/run_slepc_slicing.py \
  --data-root /home/esul01/data \
  --out-dir   /home/esul01/data/outputs/slepc_slicing_triaxial \
  --a 1.0 --b 1.5 --c 2.3 --mesh-h 0.06 --order 2 \
  --lambda-lo 0.0 --lambda-hi 449.1 \
  --partitions 8 --tol 1e-10

echo "finished=$(date)"
