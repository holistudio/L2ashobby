# Gymnasium

## Installs

Conda environment:

```
conda create --name learn-rl python=3.12
```

Install PyTorch 2.8.0 with CUDA 12.6

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Connect Four uses `pettingzoo[classic]` which contains some other games that require `open_spiel` (which is [complicated to install on Windows](https://github.com/google-deepmind/open_spiel/blob/master/docs/windows.md)). The easiest solution is to install `pettingzoo[classic]` without the `open_spiel` dependency

```
pip install pettingzoo[classic] --no-deps
```

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


