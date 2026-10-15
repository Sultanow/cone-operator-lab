#!/bin/bash
# One-time setup (login node) of the exact algebra tools for the symmetric uniqueness proof:
# Singular (deterministic Groebner bases over Q) from conda-forge, and msolve built from source against the same
# FLINT/GMP/MPFR. Afterwards:  export CT_SINGULAR=$HOME/envs/ct-algebra/bin/Singular  CT_MSOLVE=$HOME/opt/msolve/bin/msolve
set -euo pipefail
source "$(dirname "$0")/config.sh"
source "$CT_CONDA_SH"
ENV=$HOME/envs/ct-algebra
conda create -y -p "$ENV" -c conda-forge singular libflint gmp mpfr autoconf automake libtool make compilers git
conda activate "$ENV"
Singular -v -q -c 'quit;' >/dev/null 2>&1 && echo "Singular ok: $(command -v Singular)"
SRC=$HOME/opt/msolve-src; rm -rf "$SRC"; git clone --depth 1 https://github.com/algebraic-solving/msolve.git "$SRC"
cd "$SRC" && ./autogen.sh
./configure --prefix="$HOME/opt/msolve" CPPFLAGS="-I$ENV/include" LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
make -j8 && make install
printf 'x,y\n0\nx^2+y^2-25,\nx-y-1\n' > /tmp/msolve_test.ms
"$HOME/opt/msolve/bin/msolve" -f /tmp/msolve_test.ms -o /tmp/msolve_test.out && cat /tmp/msolve_test.out   # expect (-3,-4) and (4,3)
