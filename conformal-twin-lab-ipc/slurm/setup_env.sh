#!/bin/bash
# One-time environment setup (login node). Versions are the ones tested locally.
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "$CT_CONDA_SH"
conda create -y -p "$CT_ENV" python=3.12
conda activate "$CT_ENV"
pip install "numpy==2.4.4" "scipy==1.17.1" "jax[cpu]==0.11.2" "python-flint==0.9.0" \
            "matplotlib==3.10.8" "networkx==3.6.1" "shapely==2.2.0" "cvxpy==1.9.3"
python -c "import numpy, scipy, jax, flint, matplotlib, networkx, shapely, cvxpy; print('ok', numpy.__version__, scipy.__version__, jax.__version__, flint.__version__, matplotlib.__version__)"
