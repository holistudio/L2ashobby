# Gymnasium

## Installs

Conda environment:

```
conda create --name learn-rl python=3.11
```

Install PyTorch 2.8.0 with CUDA 12.6

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

```
conda install -c conda-forge compilers
```

OR on Windows:

```
conda install -c conda-forge m2w64-toolchain
```
then

```
set CMAKE_GENERATOR=MinGW Makefiles
set CMAKE_MAKE_PROGRAM=mingw32-make.exe
set CXX=g++
set CC=gcc
```

https://visualstudio.microsoft.com/downloads/?q=build+tools#build-tools-for-visual-studio-2022

Install swig separately

```
pip install swig
```

Install requirements

```
pip3 install -r requirements.txt
```

Install `mpi4py`:
- If using Windows, use `python -m pip install mpi4py impi-rt`


