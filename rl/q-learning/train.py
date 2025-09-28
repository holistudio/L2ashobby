from tictactoe import TicTacToe
from random_agent import RandomAgent
from q_agent import QLearningAgent

EPISODES = 3



def main():

    environment = TicTacToe()

    # agent = RandomAgent()
    agent = QLearningAgent()

    for e in range(EPISODES):
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_terminal = []

        state, terminal, rewards = environment.reset()

        terminal = False

        while not terminal:
            # record state
            episode_states.append(state)

            # agent plays a piece
            action = agent.action(state)
            episode_actions.append(action)

            # move environment a step
            state, terminal, rewards = environment.step(action)
            episode_rewards.append(rewards)
            episode_terminal.append(terminal)

        # update agent experience
        agent.update_experience(episode_states, episode_actions, episode_rewards, episode_terminal)

        # update agent policy
        agent.update_policy()

    environment.score_stats()

if __name__ == "__main__":
    main()