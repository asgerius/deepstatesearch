# VERY GOOD PAPER NAME

This repository contains the code used to produce the results for the paper "VERY GOOD PAPER NAME".

## Environment

To get started, run the following to install the necessary Python dependencies and compile the C modules:

```
pip install -r requirements.txt
make
```

The code was run with Python 3.9.11, but should work for any 3.9+ version of Python.

The C code was compiled with gcc 10.3.0.
The repository also contains some legacy CUDA code.
This is not used in the present form, but was compiled with CUDA 11.5.1.
If the nvcc compiler is not installed on compilation, the CUDA code is ignored, but is otherwise compiled.

The code is build to run on Linux and has only ever been run on this.
This is especially due to the interface between Python and C, which relies on compiling the C code to shared object files, the functions in which can then be called with Python's `ctypes` module.
Running the code on Windows or MacOS will therefore most likely require some tinkering with the compiler settings.
