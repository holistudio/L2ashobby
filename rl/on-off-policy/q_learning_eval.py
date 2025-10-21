# Initialize Q(s, a) arbitrarily
# Let π(a|s) be the target policy to evaluate (fixed)
# Let μ(a|s) be the behavior policy used to collect data (can be exploratory)

for episode in range(num_episodes):

    s = env.reset()
    done = False

    while not done:
        # Choose action according to behavior policy μ
        a = sample_from_policy(mu, s)

        # Take the action, observe next state and reward
        s_next, r, done, _ = env.step(a)

        # --- Q-learning update (off-policy TD control/evaluation) ---
        # Target uses *greedy* or *target policy* action, not the actual one taken
        a_target = sample_from_policy(pi, s_next)  # use π for evaluation target

        Q[s, a] = Q[s, a] + α * (r + γ * Q[s_next, a_target] - Q[s, a])

        # Move to next state
        s = s_next
