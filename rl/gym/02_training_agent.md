# Training an Agent

Link: https://gymnasium.farama.org/introduction/train_agent/


## Common Training Issues and Solutions

### Agent Never Improves

Symptoms: Reward stays constant, large training errors Causes: Learning rate too high/low, poor reward design, bugs in update logic Solutions:

 - Try learning rates between 0.001 and 0.1
 - Check that rewards are meaningful (-1, 0, +1 for Blackjack)
 - Verify Q-table is actually being updated

### Unstable Training

Symptoms: Rewards fluctuate wildly, never converge Causes: Learning rate too high, insufficient exploration Solutions:

 - Reduce learning rate (try 0.01 instead of 0.1)
 - Ensure minimum exploration (final_epsilon ≥ 0.05)
 - Train for more episodes

### Agent Gets Stuck in Poor Strategy

Symptoms: Improvement stops early, suboptimal final performance Causes: Too little exploration, learning rate too low Solutions:

 - Increase exploration time (slower epsilon decay)
 - Try higher learning rate initially
 - Use different exploration strategies (optimistic initialization)

### Learning Too Slow

Symptoms: Agent improves but very gradually Causes: Learning rate too low, too much exploration Solutions:

 - Increase learning rate (but watch for instability)
 - Faster epsilon decay (less random exploration)
 - More focused training on difficult states


## About the Environment: Blackjack

**Clear rules**: Get closer to 21 than the dealer without going over

**Observation state**: `(player_sum, dealer_card, usable_ace)`
 - `player_sum` player's hand total
 - `dealer_card` dealer's face up car
 - `usable_ace` `True`/`False` whether the player has an ace that can be 1 or 11

**Actions**: 0 - stay, 1 - hit

**Rewards**: +1 for win, -1 for loss, 0 for draw

**Episode terminates**: When agent stands or busts (goes over 21)


## Building a Q-Learning Agent

Notice `temporal_difference = target - self.q_values[obs][action]`

OR: `temporal_difference = target - current_Q`

## Executing an action

```
observation, reward, terminated, truncated, info = env.step(action)
```

`reward`: Immediate feedback for that action
`truncated`: Whether the episode ended before it was supposed to (not for agent)
`info`: Debugging info


## Training the Agent

Note the wrapper: `env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)`

## Testing Your Trained Agent

Store the current epsilon in case you want to train later: `old_epsilon = agent.epsilon`

Set `agent.epsilon = 0.0`

At the end of testing set back/restore epsilon: `agent.epsilon = old_epsilon`

## Switch to CartPole

Change `env = gym.make("CartPole-v1")`

Observation and action space have changed
 - `env.observation_space` is `Box(low=[-4.8 -inf -0.41887903 -inf], high=[4.8 inf 0.41887903 inf], (4,), float32)`
 - `env.observation_space.shape` is `(4,)`
 - observation state is now a numpy array and can't be used as a dictionary key
 - convert those to tuples via `obs = tuple(obs)`
 - `env.action_space` is `Discrete(2)` with `env.action_space.start=0`
     - meaning it's a set `{0,1}`
     - same as BlackJackAgent example

Code runs but issues remain

 - Test Loop reports a "win-rate" which doesn't make sense for CartPole
 - Reward seems to peak early and then keeps dropping till Average Reward = 10 and Average Episode Length = 10 
    - Does that make sense for Q-Learning?
    - Revisit the common problems and solutions above.