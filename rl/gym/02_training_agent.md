# Training an Agent

## About the Environment: Blackjack

**Clear rules**: Get closer to 21 than the dealer without going over

**Observation state**: `(player_sum, dealer_card, usable_ace)`
 - `player_sum` player's hand total
 - `dealer_card` dealer's face up car
 - `usable_ace` `True`/`False` whether the player has an ace that can be 1 or 11

**Actions**: 0 - stay, 1 - hit

**Rewards**: +1 for win, -1 for loss, 0 for draw

**Episode terminates**: When agent stands or busts (goes over 21)
