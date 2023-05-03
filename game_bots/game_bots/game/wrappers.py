import torch
import numpy as np

from julia import MyGame2048

moves_mapping = {0: "l", 1: "u", 2: "r", 3: "d"}


class DQNGame:
    def __init__(self):
        self.game = MyGame2048.init_game()
        self._make_move_func = MyGame2048.make_move
        self._game_func = MyGame2048.Game
        self._vector = MyGame2048.vector
        self._is_move_available = MyGame2048.is_move_available

    def game_over(self):
        for move in range(4):
            if self.is_move_available(move):
                return False
        return True

    def is_move_available(self, move):
        return self._is_move_available(move, self.game.board)

    def make_move(self, action):
        return self._make_move_func(self.game, moves_mapping[action])

    @property
    def score(self):
        return self.game.reward

    @property
    def state(self):
        return self.game.board

    def vector(self):
        return torch.tensor(self._vector(self.game.board), dtype=torch.float32)


class TDGame:
    def __init__(self):
        self.score = 0
        self.board = MyGame2048.init_game(True).board
        self.pre_move_func = MyGame2048.pre_move
        self.Game_func = MyGame2048.Game
        self.add_tile_func = MyGame2048.add_tile
        self._is_move_available = MyGame2048.is_move_available
        self._make_move_func = MyGame2048.make_move

    def game_over(self):
        self.game = self.Game_func(self.board, 0)
        for move in range(4):
            if self.is_move_available(move):
                return False
        return True

    def is_move_available(self, move):
        return self._is_move_available(move, self.game.board)

    def __str__(self):
        board = 2**self.board
        board[board == 1] = 0
        return (
            str(board)
            + f"\n score = {str(self.score)} reached {1 << np.max(self.board)}"
        )

    def new_tile(self):
        self.board = self.add_tile_func(self.Game_func(self.board, 0), True).board

    def pre_move(self, board, score, direction):
        move = moves_mapping[direction]
        game, valid = self.pre_move_func(self.Game_func(board, score), move, True)
        copy_board = game.board
        return copy_board, game.reward, valid
