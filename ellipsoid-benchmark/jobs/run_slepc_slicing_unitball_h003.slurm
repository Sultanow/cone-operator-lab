#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=2
#SBATCH --mem=700G
#SBATCH --job-name=slepc-slicing-unitball-h003
#SBATCH --time=36:00:00
#SBATCH --output=/home/esul01/data/outputs/logs/slepc_triaxial_slicing_%j.out
#SBATCH --error=/home/esul01/data/outputs/logs/slepc_triaxial_slicing_%j.err

# Inertia-based spectrum slicing (SLEPc + MUMPS) on the SAME matrices
# as the FEM-ARPACK, block-Hankel, and certification runs.
#
# Runs under MPI with spectrum partitioning: 32 ranks are split into 8
# partitions of 4 ranks each, so eight slices are processed concurrently
# and the MUMPS factorization is distributed within each partition.
# Every rank assembles the full SciPy matrices redundantly but hands only
# its own row block to PETSc.
#
# Requires PETSc/SLEPc built with 64-bit indices: at h=0.03 the factors
# hold about 6.3e9 entries, well beyond the 2^31 limit of a 32-bit build,
# which manifests as a segmentation fault mid-run.
#
# Environment:
#   conda activate /home/esul01/envs/slepc64

source /home/jupyter/anaconda3/etc/profile.d/conda.sh
conda activate /home/esul01/envs/slepc64

cd /home/esul01/hearing-ellipsoid-bench
mkdir -p /home/esul01/data/outputs/logs

export PETSC_DIR=/home/esul01/opt/petsc-i64
export SLEPC_DIR=/home/esul01/opt/slepc-i64
unset PETSC_ARCH
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

mpirun -n ${SLURM_NTASKS} python scripts/run_slepc_slicing.py \
  --data-root /home/esul01/data \
  --out-dir   /home/esul01/data/outputs/slepc_slicing_unitball_h003 \
  --a 1.0 --b 1.0 --c 1.0 --mesh-h 0.03 --order 2 \
  --lambda-lo 0.0 --n-target 2050 --n-tiles 64 \
  --tol 1e-10 --local-nev 120 --local-ncv 300 \
  --partitions 2 \
  -st_type sinvert \
  -st_ksp_type preonly \
  -st_pc_type cholesky \
  -st_pc_factor_mat_solver_type mumps \
  -st_mat_mumps_icntl_13 1 \
  -st_mat_mumps_icntl_14 200 \
  -st_mat_mumps_icntl_4 2

echo "finished=$(date)"
