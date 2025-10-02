from tictactoe import TicTacToe
from random_agent import RandomAgent
from q_agent import QLearningAgent

import copy

EPISODES = 1000



def main():

    environment = TicTacToe(score_file='train_win-loss-draw.csv')

    # agent = RandomAgent()
    agent = QLearningAgent(decay_rate=0.0002)
    # agent = QLearningAgent(alpha=1.0, gamma=1.0)

    n_act = 0 # track total number of actions

    for ep in range(EPISODES):

        state, terminal, rewards = environment.reset(ep)

        terminal = False

        while not terminal:
            # agent plays a piece
            action = agent.action(state)
            # print(action,rewards)
            n_act += 1

            # move environment a step
            next_state, terminal, rewards = environment.step(action)

            # update agent experience
            agent.update_experience(copy.deepcopy(state), action, copy.deepcopy(next_state), rewards, terminal)

            # update state
            state = next_state

        # print(action,rewards)

        # update agent policy
        agent.update_policy()

    environment.score_stats(display=True)
    print(f"\nNumber of actions: {n_act}")

    agent.save_model()

if __name__ == "__main__":
    main()