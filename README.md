# Efficiently Solving Rubik’s Cube Using Deep Reinforcement Learning

This repository contains the code used to produce the results for the paper "Efficiently Solving Rubik’s Cube Using Deep Reinforcement Learning".

[![pytest](https://github.com/asgerius/deepstatesearch/actions/workflows/pytest.yaml/badge.svg?branch=master)](https://github.com/asgerius/deepstatesearch/actions/workflows/pytest.yaml)

## Environment

All experiments were run using

- Python 3.9.11
- GCC 10.3.0

To get started, install the necessary Python packages and compile the C modules.

```sh
pip install -r requirements.txt
git submodule update --init --recursive
make

# Optionally test that everything works by running the unit tests
pip install requirements-dev.txt
python -m pytest
```

The repository contains some legacy CUDA code.
This is not used in the present form, but was compiled with CUDA 11.5.1.
If the nvcc compiler is not installed at compilation, the CUDA code is ignored, but is otherwise compiled.

The code is build to run on Linux and has only ever been run on this.
This is especially due to the interface between Python and C, which relies on compiling the C code to shared object files, the functions in which are called with Python's `ctypes` module.
Running the code on Windows or Mac will therefore probably require some tinkering with the compiler settings.

## Train models

To train the models listed in the paper, you can use the following commands.
The models were originially trained on two Intel Xeon Gold 6226R in a dual-socket setup with a Nvidia A100.
As such, the code is not set up to take advantage of multiple GPU's.
With this configuration, training a single model usually took between one and three days, depending on model size and environment.

```sh
# How many threads to use for parallel execution. In most cases,
# this should equal the number of threads on your CPU. More is
# generally better, but there are significantly diminishing returns,
# as most of the work is done on the GPU, and not all CPU work is parallel.
export OMP_NUM_THREADS=1

# Train the models
OUT=path/to/put/results
python deepstatesearch/train/run.py $OUT -c configs/Dss-Base.ini
python deepstatesearch/train/run.py $OUT -c configs/Dss-Large.ini
python deepstatesearch/train/run.py $OUT -c configs/Dss-Small.ini
python deepstatesearch/train/run.py $OUT -c configs/Dss-Smallx2.ini
python deepstatesearch/train/run.py $OUT -c configs/DeepCubeA-R.ini
python deepstatesearch/train/run.py $OUT -c configs/Dss-SP15.ini
python deepstatesearch/train/run.py $OUT -c configs/Dss-SP24.ini

# Evaluate a model
# Modify the config file at configs/eval.ini to set the hyperparameters that will be tested
MODEL=DSS_BASE

python deepstatesearch/eval/run.py  \
    $OUT/$MODEL                     \
    -c configs/eval.ini

# Evaluate a Rubik's cube model on 3,341 states exactly distance 24 from the solved state
python deepstatesearch/eval/run.py  \
    $OUT/$MODEL                     \
    -c configs/eval.ini             \
    --cube-data-file assets/qtm.txt

# Evaluation plots
# The job names should correspond to the different evaluation sections in configs/eval.ini
python deepstatesearch/plot/plot_eval.py $OUT/$MODEL LAMBDA06 LAMBDA07 LAMBDA08
```
