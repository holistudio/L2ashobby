from tictactoe import TicTacToe
from random_agent import RandomAgent
from q_agent import QLearningAgent

import datetime
import copy

EPISODES = 100000



def main():

    environment = TicTacToe(score_file='train_win-loss-draw.csv')

    # agent1 = RandomAgent()
    agent1 = QLearningAgent(decay_rate=0.0002)
    # agent1 = QLearningAgent(alpha=1.0, gamma=1.0)

    # agent2 = RandomAgent()
    # agent2 = QLearningAgent(decay_rate=0.0002)
    # agent2 = QLearningAgent(alpha=1.0, gamma=1.0)
    agent2 = agent1

    n_act = 0 # track total number of actions

    start_time = datetime.datetime.now()
    for ep in range(EPISODES):

        state1, terminal, rewards = environment.reset(ep)

        terminal = False

        while not terminal:
            # print(f'state1: {state1}')

            # agent plays a piece
            action1 = agent1.action(state1)
            n_act += 1

            # move environment a step
            next_state2, terminal, rewards = environment.step(action1)
            # print(f'     state1: {state1}')
            # print(f'next_state2: {next_state2}')
            # print(f'  terminal1: {terminal}')

            if n_act > 1:
                # update agent experience
                # print(f'     agent2')
                # print(f'     state2: {state2}')
                # print(f'next_state2: {next_state2}')
                agent2.update_experience(copy.deepcopy(state2), action2, copy.deepcopy(next_state2), rewards, terminal)
            
            if not terminal:
                # update state
                state2 = copy.deepcopy(next_state2)
                # print(f'state2: {state2}')

                # agent plays a piece
                action2 = agent2.action(state2)
                n_act += 1

                # move environment a step
                next_state1, terminal, rewards = environment.step(action2)
                # print(f'     state2: {state2}')
                # print(f'next_state1: {next_state1}')
                # print(f'  terminal2: {terminal}')

                if not terminal:
                    # update agent experience
                    # print(f'     agent1')
                    # print(f'     state1: {state1}')
                    # print(f'next_state1: {next_state1}')
                    agent1.update_experience(copy.deepcopy(state1), action1, copy.deepcopy(next_state1), rewards, terminal)
                    # update state
                    state1 = copy.deepcopy(next_state1)
                else:
                    # if the game is over after agent2's action
                    # agent2 still needs to record that as experience
                    # print(f'     agent2')
                    # print(f'     state2: {state2}')
                    # print(f'next_state2: {next_state1}')
                    agent2.update_experience(copy.deepcopy(state2), action2, copy.deepcopy(next_state1), rewards, terminal)
            else:
                # if the game is over after agent1's action
                # agent1 still needs to record that as experience
                # print(f'     agent1')
                # print(f'     state1: {state1}')
                # print(f'next_state2: {next_state2}')
                agent1.update_experience(copy.deepcopy(state1), action1, copy.deepcopy(next_state2), rewards, terminal)

        # update agent policy
        agent1.update_policy()
        agent2.update_policy()
        if (ep+1) % 10000 == 0:
            print(ep+1, datetime.datetime.now()-start_time)

    environment.score_stats(display=True)
    print(f"\nNumber of actions: {n_act}")

    agent1.save_model()
    
if __name__ == "__main__":
    main()