from copy import deepcopy
from typing import Tuple
from julia import MyGame2048

import numpy as np

SPM_SCALE = MyGame2048.SPM_SCALE
SL_SCALE = MyGame2048.SL_SCALE
SEARCH_PARAM = MyGame2048.SEARCH_PARAM
NUMBER_OF_MOVES = MyGame2048.NUMBER_OF_MOVES
MAX_VALUE = MyGame2048.MAX_VALUE


class HeuristicBot:
    def __init__(self) -> None:
        self.game = MyGame2048.init_game()
        self._make_move_func = MyGame2048.make_move
        self._random_move_func = MyGame2048.random_move
        self._game_func = MyGame2048.Game
        self.moves = []

    def make_move(self, board: np.ndarray, move: str) -> bool:
        game, is_valid = self._make_move_func(
            self._game_func(board, self.game.reward), move
        )
        return game, is_valid

    def random_move(self, board: np.ndarray) -> bool:
        game, is_valid = self._random_move_func(
            self._game_func(board, self.game.reward)
        )
        return game, is_valid

    def play(self):
        move_number = 0
        while True:
            move_number += 1
            self.game, is_valid = self.__play(move_number)
            if not is_valid:
                break
            if MAX_VALUE in self.game.board:
                break
            if move_number == 1500:
                break
        print("move number", move_number)

    def __play(self, move_number: int):
        possible_first_moves = ["l", "r", "u", "d"]
        number_of_simulations, search_length = self.get_search_params(move_number)
        first_move_scores = np.zeros(NUMBER_OF_MOVES)

        old_board = deepcopy(self.game.board)
        for first_move_index in range(NUMBER_OF_MOVES):
            move = possible_first_moves[first_move_index]
            game, is_valid = self.make_move(old_board, move)
            if not is_valid:
                continue
            for _ in range(number_of_simulations):
                move_number = 1
                search_board = np.copy(game.board)
                is_valid = True
                while move_number < search_length:
                    new_game, is_valid = self.random_move(search_board)
                    if is_valid:
                        search_board = new_game.board
                        first_move_scores[first_move_index] += new_game.reward
                    move_number += 1

        best_move_index = np.argmax(first_move_scores)
        best_move = possible_first_moves[best_move_index]
        best_game, is_valid = self.make_move(old_board, best_move)
        return best_game, is_valid

    def get_search_params(self, move_number: int) -> Tuple[int, int]:
        searches_per_move = SPM_SCALE * (1 + (move_number // SEARCH_PARAM))
        search_length = SL_SCALE * (1 + (move_number // SEARCH_PARAM))
        return searches_per_move, search_length


import time

HeuristicBotJl = MyGame2048.HeuristicBot
play = MyGame2048.play
start = time.time()
HeuristicBotJl = HeuristicBotJl()

print(play(HeuristicBotJl).board)
print(time.time() - start)
start = time.time()

bot = HeuristicBot()
bot.play()

print(bot.game.board)
print(time.time() - start)
