import random
import math

class QLearningAgent(object):
    def __init__(self, eps = 1.0, min_eps=0.1, max_eps=1.0, decay_rate=0.01):
        self.q_lookup = {}
        # snippet
        # q_lookup = {
        #     ...
        #     'X,O,X;_,_,_;_,_,_;': [0,0,0,0,0,0,0,0,0],
        #     ...
        # }

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
    
    def state_to_key(self, state):
        # convert state into a string for Q-lookup table row/key
        state_key = ""
        for row in state:
            for i,item in enumerate(row):
                if item == " ":
                    state_key += "_"
                else:
                    state_key += item
                if i < len(row)-1:
                    state_key += ","
                else:
                    state_key += ";"
        return state_key
    
    def explore(self, ixs):
        # randomly select among the indices
        return random.choice(ixs)
    
    def exploit(self, state_key, ixs):
        if state_key in self.q_lookup.keys():
            # if state exists in the lookup table
            # check values in the table row
            row = self.q_lookup[state_key]

            # find the max Q-value
            max_q = max(row)

            if row.count(max_q) == 1:
                # if there is only one action
                # get the index of the max Q-value
                ix = row.index(max_q)
            else:
                # if there is more than one action with max-Q-value
                # randomly select among those actions
                max_ixs = [i for i,v in enumerate(row) if v==max_q]
                ix = self.explore(max_ixs)
        else:
            # if new state not in the lookup table
            # explore!
            ix = self.explore(ixs)

        # TODO: ensure that all 9 action keys are added to the lookup table for encountered state
        return ix
    

    def action(self, state):
        def ix_to_loc(ix):
            # convert board index into a r,c location
            r = ix // 3
            c = ix % 3
            return (r,c)
        
        # parse state into a lookup key
        state_key = self.state_to_key(state)

        # identify blank locations on the board
        board_flat = [e for row in state for e in row]
        # print(board_flat)
        
        blank_ixs = [i for i,e in enumerate(board_flat) if e == " "]
        # print(blank_ixs)

        if random.random() < self.eps:
            # randomly select among the remaining blank locations
            select_ix = self.explore(blank_ixs)
        else:
            # follow Q-lookup table
            select_ix = self.exploit(state_key, blank_ixs)
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
        # TODO: update Q-lookup table based on experience

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