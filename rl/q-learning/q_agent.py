import random
import math
import json
class QLearningAgent(object):
    def __init__(self, eps = 1.0, min_eps=0.1, max_eps=1.0, decay_rate=0.01, alpha=0.1, gamma=0.8, 
                 load=False, model_file='q-table.json'):
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

        self.alpha = alpha
        self.gamma = gamma

        self.model_file = model_file
        if load:
            self.load_model(model_file)
        pass

    def load_model(self, load_file):
        print('\n** LOADING MODEL **')
        with open(load_file,"r") as f:
            self.q_lookup = json.load(f)
        print(f"States observed: {len(list(self.q_lookup.keys()))}")
        pass

    def save_model(self):
        print('\n** SAVING MODEL **')
        with open(self.model_file,"w") as f:
            json.dump(self.q_lookup, f)
        pass

    def epsilon_decay(self):
        # decrease epsilon based on steps
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-self.decay_rate * self.step)

        self.step += 1
        return self.epsilon
    
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

        if random.random() < self.epsilon:
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
    
    def update_experience(self, states, actions, next_states, rewards, terminals):
        # print()
        # print("# UPDATING EXPERIENCE #")
        # print(len(states), len(actions), len(next_states), len(rewards), len(terminals))
        for exp in zip(states, actions, next_states, rewards, terminals):
            state, action, next_state, reward, terminal = exp
            # print(state, action, reward, terminal)
            self.experience.append({
                "state": state,
                "action": action,
                "next_state": next_state,
                "rewards": reward,
                "terminal": terminal
            })
        pass

    def loc_to_ix(self, loc):
        r,c = loc
        ix = r*3 + c
        return ix
    
    def update_policy(self):
        """
        update Q-lookup table based on experience in a single episode
        """
        # print()
        # print("## UPDATING POLICY ##")
        # loop through experience
        for i in range(len(self.experience)):
            # track the turn index and identify the appropriate player
            if i % 2 == 0:
                pix = 0
            else:
                pix = 1

            # get state action and reward for the appropriate player
            state, action = self.experience[i]["state"], self.experience[i]["action"]
            reward = self.experience[i]["rewards"][pix]
            if reward == None:
                reward = 0.0

            # convert state to lookup table key
            state_key = self.state_to_key(state)
            # print(state_key)

            # convert (r,c) action into index
            action_i = self.loc_to_ix(action)

            # print(state, action, action_i, reward)

            # get old q-value from q_lookup table
            if not (state_key in self.q_lookup.keys()):
                # if never encountered state, initialize
                self.q_lookup[state_key] = [0 for i in range(9)]

            old_q = self.q_lookup[state_key][action_i]

            # if it's not the end of the experience (guaranteed by ending at len(experience)-1))
            # get the next state and next max q-value
            next_state =  self.experience[i]["next_state"]
            next_state_key = self.state_to_key(next_state)

            if not (next_state_key in self.q_lookup.keys()):
                # if never encountered state, initialize
                self.q_lookup[next_state_key] = [0 for i in range(9)]
            
            next_max = max(self.q_lookup[next_state_key])

            # update new value
            new_q = (1 - self.alpha) * old_q + self.alpha * (reward + self.gamma * next_max)

            self.q_lookup[state_key][action_i] = new_q
        
        # print(self.experience[i]["next_state"])
        # print()
        # print("Q-table")
        # for key in self.q_lookup:
        #     print(f"state: {key} | {self.q_lookup[key]}")

        # after learning, clear experience
        self.experience = []
        pass

def test():
    agent = QLearningAgent()

    # board = [[" "," "," "],[" "," "," "],[" "," "," "]]
    # print(agent.action(board))

    board = [["X"," ","O"],["X","X","O"],["O"," ","X"]]
    # print(agent.action(board))
    print(agent.state_to_key(board))

if __name__ == '__main__':
    test()