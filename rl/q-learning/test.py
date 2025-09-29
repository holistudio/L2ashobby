from tictactoe import TicTacToe
from random_agent import RandomAgent
from q_agent import QLearningAgent

import copy

EPISODES = 1000



def main():

    environment = TicTacToe()

    # agent1 = RandomAgent()
    agent1 = QLearningAgent(load=True)

    # agent2 = RandomAgent()
    # agent2 = QLearningAgent(load=True)
    agent2 = agent1

    for e in range(EPISODES):
        episode_states = []
        episode_next_states = []
        episode_actions = []
        episode_rewards = []
        episode_terminal = []

        state, terminal, rewards = environment.reset()

        terminal = False

        while not terminal:
            # record state
            episode_states.append(copy.deepcopy(state))

            # agent 1 or 2 plays a piece
            if environment.turn % 2 == 0:
                action = agent1.action(state)
            else:
                action = agent2.action(state)
            episode_actions.append(action)
            # print(action,rewards)

            # move environment a step
            state, terminal, rewards = environment.step(action)
            episode_next_states.append(copy.deepcopy(state))
            episode_rewards.append(rewards)
            episode_terminal.append(terminal)

        # print(action,rewards)

        # update agent experience
        # agent.update_experience(episode_states, episode_actions, episode_next_states, episode_rewards, episode_terminal)

        # update agent policy
        # agent.update_policy()

    environment.score_stats()

    # agent.save_model()

if __name__ == "__main__":
    main()