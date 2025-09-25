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

    def step(self, action):
        """
        action: a board location to place the next piece

        (action can come from a human or AI)
        """
        # get current piece

        # check if the action is valid

        # place the piece where specified

        # check if a player has won or not
        
        # check if the game ends in a draw

        # update turn index