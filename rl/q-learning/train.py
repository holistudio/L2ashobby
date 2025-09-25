from tictactoe import TicTacToe
from random_agent import RandomAgent

def main():
    environment = TicTacToe()

    agent = RandomAgent()

    state, terminal, reward = environment.reset()

    terminal = False

    while not terminal:
        # agent plays a piece
        action = agent.action(state)

        # move environment a step
        state, terminal, reward = environment.step(action)


if __name__ == "__main__":
    main()