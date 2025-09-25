class TicTacToe(object):
    def __init__(self):
        # initial empty board
        self.state = [[" "," "," "],[" "," "," "],[" "," "," "]]

        # possible board pieces
        self.pieces = ('X', 'O')

        self.turn = 0 # turn index

        # track if game is over or not
        self.terminal = False

        # player scores
        self.scores = (0, 0) # player 1 and 2 start assuming a draw
        # ( 1, -1): player 1 wins
        # (-1,  1): player 2 wins
        # ( 0,  0): draw