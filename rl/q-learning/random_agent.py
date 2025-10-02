import random

class RandomAgent(object):
    def __init__(self):
        # stores experience in the current game
        self.experience = []
        pass

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
    
    def update_experience(self, state, action, next_state, rewards, terminal):
        # print()
        # print("# UPDATING EXPERIENCE #")
        # print(state, action, reward, terminal)
        self.experience.append({
            "state": state,
            "action": action,
            "next_state": next_state,
            "rewards": rewards,
            "terminal": terminal
        })
        pass

    def update_policy(self):
        # learns nothing!

        # after learning, clear experience
        self.experience = []
        pass

    def save_model(self):
        print('\n** SAVING MODEL **')
        with open('joke.txt',"w") as f:
            f.write('I KNOW NOTHING\nI NEVER LEARN\nI LEARN NOTHING\nI NEVER KNOW')
        pass
        
def test():
    agent = RandomAgent()

    # board = [[" "," "," "],[" "," "," "],[" "," "," "]]
    # print(agent.action(board))

    board = [["X"," ","O"],["X","X","O"],["O"," ","X"]]
    print(agent.action(board))

if __name__ == '__main__':
    test()