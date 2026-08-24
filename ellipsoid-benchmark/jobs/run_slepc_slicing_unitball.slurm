#!/bin/bash
#SBATCH --job-name=slepc-slicing-unitball
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=360G
#SBATCH --time=36:00:00
#SBATCH --output=%x_%j.out

# Inertia-based spectrum slicing (SLEPc + MUMPS) for the UNIT BALL, on
# the SAME matrices as the FEM-ARPACK and block-Hankel unit-ball runs.
#
# The unit ball is the sharper test of this comparison: degeneracies
# reach 2l+1 ~ 55 in the resolved interval, and slicing guarantees
# correct multiplicities via inertia, so the recovered multiplicities
# can be checked directly against the exact spherical Bessel spectrum.
#
# --lambda-hi matches the top tile edge of the block-Hankel unit-ball
# run (1018.0667), so both methods cover the identical interval.
# --lambda-lo 0 is admissible: the smallest eigenvalue is pi^2 ~ 9.87.
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
  --out-dir   /home/esul01/data/outputs/slepc_slicing_unitball \
  --a 1.0 --b 1.0 --c 1.0 --mesh-h 0.06 --order 2 \
  --lambda-lo 0.0 --lambda-hi 1018.1 \
  --partitions 8 --tol 1e-10

echo "finished=$(date)"
