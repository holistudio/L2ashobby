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

    def get_piece(self):
        """
        Alternate between X or O piece depending on the turn index
        """
        if self.turn % 2 == 0:
            return self.pieces[0]
        else:
            return self.pieces[1]
    
    def valid_action(self, loc):
        """
        Check if the board location is empty/can receive a piece.
        """
        # get row, column coordinates
        r, c = loc

        # check if location is within bounds of the board
        if (r > 2 or r < 0) or (c > 2 or c < 0):
             return ValueError(f'Error: Board location ({r},{c}) is out of bounds.')

        # if the board location is blank, the user move is valid
        if self.board[r][c] == " ":
            return True
        else:
            return ValueError(f'Error: Board location ({r},{c}) is already filled with a piece.')
    
    def place_piece(self, piece, loc):
        # get row, column coordinates
        r, c = loc

        self.state[r][c] = piece
        return
        
    def step(self, action):
        """
        action: a board location to place the next piece

        (action can come from a human or AI)
        """
        # get current piece
        piece = self.get_piece()

        # check if the action is valid
        self.valid_action(loc=action)

        # place the piece where specified
        self.place_piece(piece=piece, loc=action)

        # check if a player has won or not

        # check if the game ends in a draw

        # update turn index