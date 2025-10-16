# --- Tabular SARSA (on-policy) ---

# Initialize Q-table arbitrarily (e.g., zeros)
Q = defaultdict(lambda: np.zeros(num_actions))

# Hyperparameters
alpha = 0.1        # learning rate
gamma = 0.99       # discount factor
epsilon = 0.1      # exploration rate

for episode in range(num_episodes):
    s = env.reset()

    # --- Choose first action using *current* policy (ε-greedy)
    if random.random() < epsilon:
        a = random_action()
    else:
        a = argmax(Q[s])

    done = False

    while not done:
        # Take the chosen action
        s_next, r, done, _ = env.step(a)

        # --- Behavior policy: ε-greedy (used to *act*)
        # --- Target policy: same ε-greedy (used to *learn*)
        # This means the update reflects what we *actually did*, not a hypothetical greedy action.
        if random.random() < epsilon:
            a_next = random_action()
        else:
            a_next = argmax(Q[s_next])

        # --- On-policy TD target: uses the *actual next action taken*
        td_target = r + gamma * Q[s_next][a_next]
        td_error = td_target - Q[s][a]

        # --- Update rule
        Q[s][a] += alpha * td_error

        # Move to next step
        s, a = s_next, a_next
