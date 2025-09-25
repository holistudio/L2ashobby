import random

class RandomAgent(object):
    def action(self, state):
        def ix_to_loc(ix):
            r = ix // 3
            c = ix % 3
            return (r,c)
        # identify blank locations on the board
        board_flat = [e for row in state for e in row]
        print(board_flat)
        blank_ixs = [i for i,e in enumerate(board_flat) if e == " "]
        print(blank_ixs)

        # randomly select among the remaining blank locations
        select_ix = random.choice(blank_ixs)
        loc = ix_to_loc(select_ix)
        return loc

def test():
    agent = RandomAgent()

    # board = [[" "," "," "],[" "," "," "],[" "," "," "]]
    # print(agent.action(board))

    board = [["X"," ","O"],["X","X","O"],["O"," ","X"]]
    print(agent.action(board))

if __name__ == '__main__':
    test()