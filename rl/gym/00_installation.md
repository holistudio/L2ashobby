# Installation

Installation is not covered in the main website: https://gymnasium.farama.org/

Refer to the GitHub repo instead: https://github.com/Farama-Foundation/Gymnasium

Default install: `pip install gymnasium`

BUT there's probably more for RL training on specific games

Biggest install: `pip install "gymnasium[all]"`

HOWEVER, this seems to result in a build error during:

```
Building wheels for collected packages: box2d-py
  DEPRECATION: Building 'box2d-py' using the legacy setup.py bdist_wheel mechanism, which will be removed in a future version. pip 25.3 will enforce this behaviour change. A possible replacement is to use the standardized build interface by setting the `--use-pep517` option, (possibly combined with `--no-build-isolation`), or adding a `pyproject.toml` file to the source tree of 'box2d-py'. Discussion can be found at https://github.com/pypa/pip/issues/6334

... more stuff
...
  ERROR: Failed building wheel for box2d-py
```

SOOO start with these installs in requirements.txt:
```
gymnasium
gymnasium[classic-control]
```