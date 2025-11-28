# Install [FEniCS](https://fenicsproject.org/)

# Conda Environment

```console
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh

export PATH="$HOME/miniconda3/bin:$PATH"
conda init bash

#bei SSL Fehlern:
conda config --set ssl_verify False

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n fenicsx-env python=3.12 -y
conda activate fenicsx-env

conda install -c conda-forge fenics-basix fenics-ffcx fenics-ufl fenics-dolfinx slepc4py petsc4py mpi4py gmsh -y

sudo apt update
sudo apt install gmsh -y
pip install gmsh
```

# Run the programs

First generate the mesh via `python make_ellipsoid_mesh.py` and then calculate the eigenvalues via `python ellipsoid_eigs.py` or `mpirun -n 1 python ellipsoid_eigs.py`.
