import random

class QLearningAgent(object):
    def __init__(self, eps = 1.0):
        # stores experience in the current game
        self.experience = []

        self.eps = eps
        self.step = 0
        pass

    def eps_decay(self):
        # decrease epsilon based on 
        # self.eps = self.step * exp()
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

        # randomly select among the remaining blank locations
        select_ix = random.choice(blank_ixs)
        loc = ix_to_loc(select_ix)
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