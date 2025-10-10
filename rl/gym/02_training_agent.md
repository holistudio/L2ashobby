# Training an Agent

Link: https://gymnasium.farama.org/introduction/train_agent/

## About the Environment: Blackjack

**Clear rules**: Get closer to 21 than the dealer without going over

**Observation state**: `(player_sum, dealer_card, usable_ace)`
 - `player_sum` player's hand total
 - `dealer_card` dealer's face up car
 - `usable_ace` `True`/`False` whether the player has an ace that can be 1 or 11

**Actions**: 0 - stay, 1 - hit

**Rewards**: +1 for win, -1 for loss, 0 for draw

**Episode terminates**: When agent stands or busts (goes over 21)


## Executing an action

```
observation, reward, terminated, truncated, info = env.step(action)
```

`reward`: Immediate feedback for that action
`truncated`: Whether the episode ended before it was supposed to (not for agent)
`info`: Debugging info

