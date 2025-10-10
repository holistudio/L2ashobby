# Basic Usage

Link: https://gymnasium.farama.org/introduction/basic_usage/

## Questions

 - What are the four values in the observation space for CartPole?
 - `OrderEnforcing`: ensures proper step order and resets...?
 - `PassiveEnvChecker`: validates environment usage...?


## Your First RL Program

NOTE:  Run `pip install "gymnasium[classic-control]"` before running the CartPole example

### Explaining the Code Step by Step

`env=gym.make()`
 - Specify the game
 - optional `render_mode="human"`
 - See [`Env.render()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.render) for details on different render modes.
 - specifying `render_mode=None` is fastest for training obviously


`observation, info = env.reset()`
 - start the game
 - get the first `observation` for what the agent can see
 - `info` for debugging stuff
 - random seed options available for `reset()`


 `episode_over` tracks terminal state of episode.

### Action and observation spaces

`Env.action_space` and `Env.observation_space` are instances of Space, a high-level python class that provides key functions: `Space.contains()` and `Space.sample()`

Lots of discrete and continuous options
 - `Box`: describes bounded space with upper and lower limits of any n-dimensional shape (like continuous control or image pixels).
 - `Discrete`: describes a discrete space where {0, 1, ..., n-1} are the possible values (like button presses or menu choices).
 - `Dict`: describes a dictionary of simpler spaces (like our GridWorld example you’ll see later).


```
import gymnasium as gym

# Discrete action space (button presses)
env = gym.make("CartPole-v1")
print(f"Action space: {env.action_space}")  # Discrete(2) - left or right
print(f"Sample action: {env.action_space.sample()}")  # 0 or 1

# Box observation space (continuous values)
print(f"Observation space: {env.observation_space}")  # Box with 4 values
# Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf])
print(f"Sample observation: {env.observation_space.sample()}")  # Random valid observation
```


## Modifying the environment

Gynasium games already have these Wrapper to allow you to modify environment behavior without touching the underlying code.

 - `TimeLimit`: stops episode after a certain number of steps
 - `OrderEnforcing`: ensures proper step order and resets...?
 - `PassiveEnvChecker`: validates environment usage...?

 New wrappers add functionality and can be chained to combine effects

 ```
# Wrap it to flatten the observation into a 1D array
wrapped_env = FlattenObservation(env)
```

Additional Wrappers:

 - `ClipAction`: Clips any action passed to step to ensure it’s within the valid action space.
 - `RescaleAction`: Rescales actions to a different range (useful for algorithms that output actions in [-1, 1] but environment expects [0, 10]).

Easily get back to completely unwrapped environment with `wrapped_env.unwrapped`

