from tictactoe import TicTacToe
from random_agent import RandomAgent

EPISODES = 3

def main():
    score_board = []

    environment = TicTacToe()

    agent = RandomAgent()

    for e in range(EPISODES):
        state, terminal, reward = environment.reset()

        terminal = False

        while not terminal:
            # agent plays a piece
            action = agent.action(state)

            # move environment a step
            state, terminal, rewards = environment.step(action)

        score_board.append(rewards)


if __name__ == "__main__":
    main()