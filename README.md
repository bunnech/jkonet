# Proximal Optimal Transport Modeling of Population Dynamics

This repository contains ...

## Installation

To install all dependencies, execute the following steps:
```
conda create --name jko python=3.9.7
conda activate jko

conda update -n base -c defaults conda

pip install -r requirements.txt
python setup.py develop
```
In case you do not use miniconda, make sure to use the right versions of the libraries
specified in the `requirements` file.

If you work on GPUs, please download `jax` with CUDA support (see [here](https://github.com/google/jax#installation)).

If you want jupyter notebook support (may have errors), run the following 
commands (inside `jko`):
```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=jko
```
Change the kernel name to `jko` or create a new ipython notebook using `jko` 
as the kernel.

## Run Experiments
...

## Contact
In case you have questions, reach out to `bunnec@inf.ethz.ch`.
