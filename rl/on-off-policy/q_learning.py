# --- Tabular Q-learning (off-policy) ---

# Initialize Q-table arbitrarily (e.g., zeros)
Q = defaultdict(lambda: np.zeros(num_actions))

# Hyperparameters
alpha = 0.1        # learning rate
gamma = 0.99       # discount factor
epsilon = 0.1      # exploration rate

for episode in range(num_episodes):
    s = env.reset()
    done = False

    while not done:
        # --- Behavior policy: ε-greedy (used to *collect* data)
        if random.random() < epsilon:
            a = random_action()
        else:
            a = argmax(Q[s])  # choose best known action

        # Take action in the environment
        s_next, r, done, _ = env.step(a)

        # --- Target policy: greedy (used to *compute* target)
        # Note: this uses the max over next actions,
        # which corresponds to a policy that *always* picks the best action
        best_next_action = argmax(Q[s_next])

        # --- Update rule (off-policy TD target)
        td_target = r + gamma * Q[s_next][best_next_action]
        td_error = td_target - Q[s][a]
        Q[s][a] += alpha * td_error

        # Move to next state
        s = s_next
