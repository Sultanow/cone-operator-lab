# Shared job prologue: config, environment, thread pinning.
CT_CONFIG=${CT_CONFIG:-${SLURM_SUBMIT_DIR:-$PWD}/slurm/config.sh}
source "$CT_CONFIG"
if [ -n "${CT_CONDA_SH:-}" ]; then
  set +u                      # conda's activation scripts are not 'set -u' safe
  source "$CT_CONDA_SH"
  conda activate "$CT_ENV"
  set -u
fi
NT=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$NT"
echo "started=$(date) node=$(hostname) threads=$NT python=$(command -v python3)"
