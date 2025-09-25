from tictactoe import TicTacToe
from random_agent import RandomAgent

EPISODES = 3

def score_stats(score_board):
    p1_stats=[0,0,0] # win-loss-draw
    p2_stats=[0,0,0]
    for result in score_board:
        if result[0] == 1:
            p1_stats[0]+=1
            p2_stats[1]+=1
        elif result[0] == -1:
            p1_stats[1]+=1
            p2_stats[0]+=1
        else:
            p1_stats[2]+=1
            p2_stats[2]+=1

    p1_win_rate = p1_stats[0]* 100/sum(p1_stats) 
    p2_win_rate = p2_stats[0]* 100/sum(p2_stats)

    print(f'Player 1 ({p1_win_rate:.2f}%): {p1_stats[0]} W - {p1_stats[1]} L - {p1_stats[2]} D ')
    print(f'Player 2 ({p2_win_rate:.2f}%): {p2_stats[0]} W - {p2_stats[1]} L - {p2_stats[2]} D ')

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

    score_stats(score_board)

if __name__ == "__main__":
    main()