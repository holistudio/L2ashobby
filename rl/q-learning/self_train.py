from tictactoe import TicTacToe
from random_agent import RandomAgent
from q_agent import QLearningAgent

import copy

EPISODES = 1



def main():

    environment = TicTacToe(display=True)

    agent1 = RandomAgent()
    # agent1 = QLearningAgent(decay_rate=0.0002)
    # agent1 = QLearningAgent(alpha=1.0, gamma=1.0)

    agent2 = RandomAgent()
    # agent2 = QLearningAgent(decay_rate=0.0002)
    # agent2 = QLearningAgent(alpha=1.0, gamma=1.0)

    n_act = 0 # track total number of actions

    for ep in range(EPISODES):

        state1, terminal, rewards = environment.reset(ep)

        terminal = False

        while not terminal:
            # agent plays a piece
            action1 = agent1.action(state1)
            # print(action,rewards)
            n_act += 1

            # move environment a step
            next_state2, terminal, rewards = environment.step(action1)

            if n_act > 1:
                agent2.update_experience(copy.deepcopy(state2), action2, copy.deepcopy(next_state2), rewards, terminal)
            
            if not terminal:
                state2 = next_state2

                # agent plays a piece
                action2 = agent2.action(state2)
                # print(action,rewards)
                n_act += 1

                # move environment a step
                next_state1, terminal, rewards = environment.step(action2)

                # update agent experience
                agent1.update_experience(copy.deepcopy(state1), action1, copy.deepcopy(next_state1), rewards, terminal)

                # update state
                state1 = next_state1

            # TODO: This may be a unneeded repeat of line 41 agent2.update_experience()
            # else:
            #     # if the game is over after agent1's action
            #     # agent2 still needs to record that as experience
            #     agent2.update_experience(copy.deepcopy(state2), action2, copy.deepcopy(next_state2), rewards, terminal)

        # print(action,rewards)

        # update agent policy
        agent1.update_policy()
        agent2.update_policy()

    environment.score_stats(display=True)
    print(f"\nNumber of actions: {n_act}")

    agent1.save_model()
    # TODO: see if all you need is agent1 save_model() to save a **complete** Q-table
    # otherwise this needs two save_model() and a combine function something something

if __name__ == "__main__":
    main()