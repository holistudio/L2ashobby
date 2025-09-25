import random

class RandomAgent(object):
    def action(self, state):
        def ix_to_loc(ix):
            r = ix // 3
            c = ix % 9
            return (r,c)
        # identify blank locations on the board
        board_flat = [e for row in state for e in row]
        blank_ixs = [i for i,e in enumerate(board_flat) if e == " "]

        # randomly select among the remaining blank locations
        select_ix = random.choice(blank_ixs)
        loc = ix_to_loc(select_ix)
        return loc
