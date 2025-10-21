# Initialize Q(s, a) arbitrarily for all states s, actions a
# Let π(a|s) be the fixed target policy we wish to evaluate

for episode in range(num_episodes):

    # Start a new episode
    s = env.reset()
    a = sample_from_policy(pi, s)   # Choose action according to π(a|s)

    done = False
    while not done:

        # Take the action, observe next state and reward
        s_next, r, done, _ = env.step(a)

        # Choose next action using the SAME policy π
        a_next = sample_from_policy(pi, s_next)

        # SARSA update (on-policy TD(0) update)
        Q[s, a] = Q[s, a] + α * (r + γ * Q[s_next, a_next] - Q[s, a])

        # Move to the next state and action
        s, a = s_next, a_next
