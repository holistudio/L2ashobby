from tictactoe import TicTacToe
from random_agent import RandomAgent

EPISODES = 3



def main():

    environment = TicTacToe()

    agent = RandomAgent()

    for e in range(EPISODES):
        state, terminal, rewards = environment.reset()

        terminal = False

        while not terminal:
            # agent plays a piece
            action = agent.action(state)

            # move environment a step
            state, terminal, rewards = environment.step(action)

        # update agent policy
        agent.update_policy()

    environment.score_stats()

if __name__ == "__main__":
    main()