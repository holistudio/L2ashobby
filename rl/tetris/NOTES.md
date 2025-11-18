# Tetris

## Environment

### Observation space

```
>>> env.observation_space
Box(0.0, 1.0, (1, 234), float32)
```


Game Grid (200 dimensions):
    The first n_cols * n_rows (e.g., 10 * 20 = 200) values represent the game board.
    0.0: An empty cell.
    1.0: A cell occupied by a placed tetromino.
    2.0: A cell occupied by the currently falling tetromino.

Game State Floats (6 dimensions):
    These are 6 floating-point values that give the agent more context about the game's state:
        Game Tick: The current game tick normalized by the maximum ticks (tick / MAX_TICKS).
        Fall Tick: A normalized counter for when the current piece will naturally fall (tick_fall / ticks_per_fall).
        Tetromino Row: The normalized row position of the current tetromino.
        Tetromino Column: The normalized column position of the current tetromino.
        Tetromino Rotation: The current rotation index (0-3) of the tetromino.
        Can Swap: A boolean (1.0 or 0.0) indicating if the hold action is available.

Tetromino Information (28 dimensions):
    This section uses one-hot encoding to represent the current, upcoming, and held tetrominoes. There are 7 unique tetromino shapes.
    Current Piece (7 dimensions): A one-hot vector indicating the shape of the current tetromino.
    Preview 1 (7 dimensions): A one-hot vector for the next tetromino in the queue.
    Preview 2 (7 dimensions): A one-hot vector for the second tetromino in the queue.
    Held Piece (7 dimensions): A one-hot vector for the tetromino in the hold slot. If the hold slot is empty, this will be all zeros.

Noise Observations (10 dimensions by default):
    The final n_noise_obs values are for adding random noise to the observations. This is a technique used in training to make the agent more robust by learning to ignore irrelevant inputs.


### Action space

```
>>> env.action_space
MultiDiscrete([7])
```

Action ID   Action	        Description
0	        No Operation	The agent does nothing, and the current tetromino continues to fall.
1	        Left	        Moves the current tetromino one column to the left.
2	        Right	        Moves the current tetromino one column to the right.
3	        Rotate	        Rotates the current tetromino 90 degrees clockwise.
4	        Soft Drop	    Moves the current tetromino one row down.
5	        Hard Drop	    Immediately places the current tetromino at the lowest possible position.
6	        Hold	        Swaps the current tetromino with the one in the hold slot.


### Rewards

Rewards are decimal values


Line Clear	+0.1 to +1.0	
A significant reward is given when one or more lines are cleared simultaneously. The reward scales with the number of lines cleared, as defined by the REWARD_COMBO constant. This is the primary incentive for the agent.

Hard Drop	+0.02 per row	
When the agent performs a "Hard Drop", it receives a small reward for each row the tetromino travels downwards. This encourages the agent to place pieces efficiently and quickly.

Rotate	+0.01	
A minor reward is given for each successful rotation. This can be seen as a shaping reward to encourage the agent to explore different piece orientations.

Invalid Action	+0.0	
Attempting an illegal move (e.g., moving into a wall or another block) results in a neutral reward of 0.0.

Other Actions	+0.0	
Actions like moving left, right, holding a piece, or performing a soft drop do not grant any immediate reward.



## Agent

- Use convolutional neural net somehow...
- Use the raw frame data for the CNN...