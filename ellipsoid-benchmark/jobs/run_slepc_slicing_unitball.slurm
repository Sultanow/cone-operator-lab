#!/bin/bash
#SBATCH --job-name=slepc-slicing-unitball
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=36:00:00
#SBATCH --output=/home/esul01/data/outputs/logs/slepc_unitball_slicing_%j.out
#SBATCH --error=/home/esul01/data/outputs/logs/slepc_unitball_slicing_%j.err

# Inertia-based spectrum slicing (SLEPc + MUMPS) for the UNIT BALL, on
# the SAME matrices as the FEM-ARPACK and block-Hankel unit-ball runs.
#
# The unit ball is the sharper test of this comparison: degeneracies
# reach 2l+1 ~ 55 in the resolved interval, and slicing guarantees
# correct multiplicities via inertia, so the recovered multiplicities
# can be checked directly against the exact spherical Bessel spectrum.
#
# The runner derives lambda_hi with the same Weyl tiling function and
# n_target=2050 used by the block-Hankel unit-ball job.  It deliberately
# enforces one MPI rank so the SciPy-to-PETSc CSR conversion stays serial.
#
# Environment (adapt to the cluster):
#   module load petsc slepc mumps    # or: conda activate slepc-env

source /home/jupyter/anaconda3/etc/profile.d/conda.sh
conda activate slepc-env

cd /home/esul01/hearing-ellipsoid-bench
mkdir -p /home/esul01/data/outputs/logs

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_DYNAMIC=FALSE
export NUMEXPR_NUM_THREADS=1

echo "started=$(date)"
echo "node=$(hostname)"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

python scripts/run_slepc_slicing_unitball.py \
  --data-root /home/esul01/data \
  --out-dir   /home/esul01/data/outputs/slepc_slicing_unitball \
  --mesh-h 0.06 --order 2 --n-target 2050 --n-tiles 64 \
  --tol 1e-10 --local-nev 120 --local-ncv 300 \
  -st_type sinvert \
  -st_ksp_type preonly \
  -st_pc_type cholesky \
  -st_pc_factor_mat_solver_type mumps \
  -st_mat_mumps_icntl_13 1

echo "finished=$(date)"
