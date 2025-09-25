class TicTacToe(object):
    def __init__(self, win_score=1, loss_score=-1):
        # initial empty board
        self.state = [[" "," "," "],[" "," "," "],[" "," "," "]]

        # possible board pieces
        self.pieces = ('X', 'O')

        self.turn = 0 # turn index

        # track if game is over or not
        self.terminal = False

        # player scores
        self.scores = (None, None)


        # ( 1, -1): player 1 wins
        # (-1,  1): player 2 wins
        # ( 0,  0): draw
        self.possible_scores = {
            "p1_win": ( win_score, loss_score),
            "p2_win": (loss_score,  win_score),
            "draw": (0, 0)
        }

    def display(self):
        """
        Display the board to the terminal
        """
        print()
        print("BOARD")
        print("=====")
        for i,row in enumerate(self.state):
            row_disp = ("|").join(row)
            print(row_disp)
            if i < 2:
                print("-----")
        print("=====")
        print()
        return
    
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
        if self.state[r][c] == " ":
            return True
        else:
            return ValueError(f'Error: Board location ({r},{c}) is already filled with a piece.')
    
    def place_piece(self, piece, loc):
        # get row, column coordinates
        r, c = loc

        self.state[r][c] = piece
        return
    
    def check_win(self):
        """
        Check connect 3 pattern in all possible variations
        """
        def check_rows(board):
            """
            Check connect 3 pattern in rows
            """
            for i in range(3):
                # print(f"Checking row {i}...")
                # check if none of the locations in the row are blank
                location_blank = (board[i][0] == " ") or (board[i][1] == " ") or (board[i][2] == " ")
                if not location_blank:
                    # check if all pieces are same for this row
                    if board[i][0] == board[i][1]:
                        if board[i][1] == board[i][2]:
                            return True
            return False
        
        def check_cols(board):
            """
            Check connect 3 pattern in columns
            """
            for i in range(3):
                # print(f"Checking col {i}...")
                # check if none of the locations in the col are blank
                location_blank = (board[0][i] == " ") or (board[1][i] == " ") or (board[2][i] == " ")
                if not location_blank:
                    # check if all pieces are same for this col
                    if board[0][i] == board[1][i]:
                        if board[1][i] == board[2][i]:
                            return True
            return False
        
        def check_diags(board):
            """
            Check connect 3 pattern in diagonals
            """
            # top left to bottom right diagonal
            # print('Checking diagonal 1...')
            # check if none of the locations in the col are blank
            location_blank = (board[0][0] == " ") or (board[1][1] == " ") or (board[2][2] == " ")
            if not location_blank:
                # check if all pieces are same for this col
                if board[0][0] == board[1][1]:
                    if board[1][1] == board[2][2]:
                        return True
            
            # top right to bottom left diagonal
            # print('Checking diagonal 2...')
            # check if none of the locations in the col are blank
            location_blank = (board[2][0] == " ") or (board[1][1] == " ") or (board[0][2] == " ")
            if not location_blank:
                # check if all pieces are same for this col
                if board[2][0] == board[1][1]:
                    if board[1][1] == board[0][2]:
                        winning_piece = board[0][2]
                        return True
            return False
        
        # immediately return True if one of the checks is met
        if check_rows(self.state):
            return True
        
        if check_cols(self.state):
            return True
        
        if check_diags(self.state):
            return True
        
        return False
    
    def check_draw(self, win):
        if not win and self.turn >= 8:
            return True
        return False
    
    def assign_scores(self, win):
        # if the game ends with win
        if win:
            # determine the latest piece and assign player scores accordingly
            if self.turn % 2 == 0:
                return self.possible_scores["p1_win"]
            return self.possible_scores["p2_win"]
        else:
            # otherwise assign draw scores
            return self.possible_scores["draw"]
    
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

        # check if the most recent move/player has won or not
        win = self.check_win()

        # check if the game ends in a draw
        self.terminal = win or self.check_draw(win)

        # assign scores to each player if the game is done
        if self.terminal:
            self.scores = self.assign_scores(win)
        else:
            # update turn index
            self.turn += 1

        self.display()
        # return state, terminal, rewards
        return self.state, self.terminal, self.scores
    
    def reset(self):
        # initial empty board
        self.state = [[" "," "," "],[" "," "," "],[" "," "," "]]

        self.turn = 0 # turn index

        # track if game is over or not
        self.terminal = False

        # player scores
        self.scores = (None, None)

        print()
        print()
        print("### NEW GAME ###")
        self.display()
        return self.state, self.terminal, self.scores