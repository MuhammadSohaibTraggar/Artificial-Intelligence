class Player:
    def __init__(self, name, symbol):
        self.name = name
        self.symbol = symbol
    
    def play(self, game_board, position):
        game_board[position] = self.symbol


class GameBoard:
    def __init__(self, size):
        self.size = size  
        self.grid = [' ' for _ in range(self.size * self.size)] 

    def show(self):
        for i in range(self.size):
            print(' | '.join(self.grid[i * self.size:(i + 1) * self.size]))
            if i < self.size - 1:
                print('-' * (self.size * 4 - 1))

    def is_filled(self):
        return ' ' not in self.grid

    def check_winner(self, player):
        symbol = player.symbol
        for i in range(self.size):
            if all(self.grid[i * self.size + j] == symbol for j in range(self.size)): 
                return True
            if all(self.grid[j * self.size + i] == symbol for j in range(self.size)): 
                return True
        if all(self.grid[i * self.size + i] == symbol for i in range(self.size)):
            return True
        if all(self.grid[(self.size - 1 - i) * self.size + i] == symbol for i in range(self.size)):
            return True

        return False


class TicTacToeGame:
    def __init__(self, size):
        self.size = size
        self.game_board = GameBoard(size)
        self.players = []

    def add_player(self, name, symbol):
        self.players.append(Player(name, symbol))

    def switch_player(self, current_player):
        return self.players[1] if current_player == self.players[0] else self.players[0]

    def start(self):
        current_player = self.players[0]
        board_size = self.size * self.size
        while not self.game_board.is_filled():
            self.game_board.show()
            try:
                move = int(input(f"{current_player.name}, choose your box (1-{board_size}): ")) - 1
                if move < 0 or move >= board_size:
                    print(f"Invalid input. Please choose a number between 1 and {board_size}.")
                    continue
                if self.game_board.grid[move] == ' ':
                    current_player.play(self.game_board.grid, move)
                    if self.game_board.check_winner(current_player):
                        self.game_board.show()
                        print(f"Congratulations, {current_player.name} wins the game!")
                        return
                    current_player = self.switch_player(current_player)
                else:
                    print("That box is already filled. Try again.")
            except ValueError:
                print("Invalid input! Please enter a number.")

        self.game_board.show()
        print("It's a draw!")

print("Welcome to Tic Tac Toe Game!")
board_size = int(input("Enter the board size (select any number): "))
tic_tac_toe_game = TicTacToeGame(board_size)
while True:
    player1_name = input("Enter name for Player 1: ")
    player1_symbol =  input("Enter symbol for Player 1 (X/O): ")

    if  player1_symbol not in ['X', 'O']:
        print("Invalid symbol. Please enter either X or O")
    else:
        print(player1_symbol)
        break 
    
player2_name = input("Enter name for Player 2: ")
player2_symbol = 'X' if player1_symbol == 'O' else 'O'



tic_tac_toe_game.add_player(player1_name, player1_symbol)
tic_tac_toe_game.add_player(player2_name, player2_symbol)
tic_tac_toe_game.start()
