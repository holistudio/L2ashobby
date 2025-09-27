import random
import math

class QLearningAgent(object):
    def __init__(self, eps = 1.0, min_eps=0.1, max_eps=1.0, decay_rate=0.01):
        # stores experience in the current game
        self.experience = []

        self.epsilon = eps
        self.min_epsilon = min_eps
        self.max_epsilon = max_eps
        self.decay_rate = decay_rate
        self.step = 0
        pass

    def epsilon_decay(self):
        # decrease epsilon based on steps
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-self.decay_rate * self.step)

        self.step += 1
        return self.eps

    def action(self, state):
        def ix_to_loc(ix):
            # convert board index into a r,c location
            r = ix // 3
            c = ix % 3
            return (r,c)
        
        # identify blank locations on the board
        board_flat = [e for row in state for e in row]
        # print(board_flat)
        
        blank_ixs = [i for i,e in enumerate(board_flat) if e == " "]
        # print(blank_ixs)

        if random.random() < self.eps:
            # randomly select among the remaining blank locations
            select_ix = random.choice(blank_ixs)
        else:
            # follow Q-table
            pass

        # convert board index into row, col location
        loc = ix_to_loc(select_ix)

        # update epsilon
        self.epsilon_decay()
        return loc
    
    def update_experience(self, states, actions, rewards, terminals):
        for exp in zip(states, actions, rewards, terminals):
            state, action, reward, terminal = exp
            self.experience.append({
                "state": state,
                "action": action,
                "rewards": reward,
                "terminal": terminal
            })
        pass

    def update_policy(self):
        # learns nothing!

        # after learning, clear experience
        self.experience = []
        pass

def test():
    agent = QLearningAgent()

    # board = [[" "," "," "],[" "," "," "],[" "," "," "]]
    # print(agent.action(board))

    board = [["X"," ","O"],["X","X","O"],["O"," ","X"]]
    print(agent.action(board))

if __name__ == '__main__':
    test()