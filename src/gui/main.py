from time import sleep
from julia import MyGame2048

from tkinter import Button, StringVar, Tk, Frame, Label
from tkinter import messagebox
import numpy as np


bg_color = {
    2: "#daeddf",
    4: "#9ae3ae",
    8: "#6ce68d",
    16: "#42ed71",
    32: "#17e650",
    64: "#17c246",
    128: "#149938",
    256: "#107d2e",
    512: "#0e6325",
    1024: "#0b4a1c",
    2048: "#031f0a",
}
color = {
    2: "#011c08",
    4: "#011c08",
    8: "#011c08",
    16: "#011c08",
    32: "#011c08",
    64: "#f2f2f0",
    128: "#f2f2f0",
    256: "#f2f2f0",
    512: "#f2f2f0",
    1024: "#f2f2f0",
    2048: "#f2f2f0",
}


class Board:
    def __init__(self):
        self.n = 4
        self.window = Tk()
        self.window.title("2048 Game")
        self.gameArea = Frame(self.window, bg="azure3")
        self.board = []
        self.gridCell = np.zeros((4, 4), dtype=np.int32)
        self.score = StringVar(value="0")
        Label(
            self.gameArea,
            text="Score:",
            font=("calibri", 13, "bold"),
        ).grid(row=0, column=1)
        Label(
            self.gameArea,
            textvariable=self.score,
            font=("calibri", 13, "bold"),
        ).grid(row=0, column=2, padx=12)
        self.button = Button(self.gameArea, text="Restart")
        self.bot_button = Button(self.gameArea, text="Bot")

        self.button.grid(row=0, column=3)
        self.bot_button.grid(row=0, column=0)

        for i in range(4):
            rows = []
            for j in range(4):
                l = Label(
                    self.gameArea,
                    text="",
                    bg="azure4",
                    font=("arial", 22, "bold"),
                    width=10,
                    height=5,
                )
                l.grid(row=i + 1, column=j, padx=1, pady=1)
                rows.append(l)
            self.board.append(rows)

        self.gameArea.grid()

    def paintGrid(self):
        for i in range(4):
            for j in range(4):
                if self.gridCell[i][j] == 0:
                    self.board[i][j].config(text="", bg="azure4")
                else:
                    cell = self.gridCell[i][j]
                    self.board[i][j].config(
                        text=str(cell),
                        bg=bg_color[cell],
                        fg=color[cell],
                    )

    def update(self, game):
        self.gridCell = game.board
        self.score.set(str(game.reward))


class Game:
    def __init__(self, board: Board):
        self.board = board
        self.board.button.configure(command=self.start)
        self.board.bot_button.configure(command=self.bot_play)

        self._make_move_func = MyGame2048.make_move
        self._game_func = MyGame2048.Game
        self._heuristic_bot = MyGame2048.HeuristicBot
        self._bot_play_func = MyGame2048.play
        self.move_number = 0

    def start(self):
        self.game = MyGame2048.init_game()
        self.board.update(self.game)
        self.board.paintGrid()
        self.board.window.bind("<Key>", self.make_move)
        self.board.window.mainloop()

    def bot_play(self):
        game = self.game

        while True:
            bot = self._heuristic_bot(game)
            self.move_number += 1
            game, is_valid = self._bot_play_func(bot, self.move_number)
            bot = self._heuristic_bot(game)
            self.board.update(game)
            self.board.paintGrid()

            if 2048 in self.board.gridCell:
                messagebox.showinfo("2048", message="You Won!!")
                self.move_number = 0
                return

            if (self.board.gridCell == 0).sum() == 0 and not is_valid:
                messagebox.showinfo("2048", "Game Over!!!")
                self.move_number = 0
                return

            if self.move_number == 1000:
                messagebox.showinfo("2048", "Move out!!!")
                self.move_number = 0
                return

            sleep(0.25)
            self.board.window.update()

    def make_move(self, event):
        presed_key = event.keysym
        if presed_key == "Up":
            self.game, is_valid = self._make_move_func(self.game, "u")
        elif presed_key == "Down":
            self.game, is_valid = self._make_move_func(self.game, "d")
        elif presed_key == "Left":
            self.game, is_valid = self._make_move_func(self.game, "l")
        elif presed_key == "Right":
            self.game, is_valid = self._make_move_func(self.game, "r")
        else:
            pass
        self.board.update(self.game)
        self.board.paintGrid()

        if 2048 in self.board.gridCell:
            messagebox.showinfo("2048", message="You Won!!")

        if (self.board.gridCell == 0).sum() == 0 and not is_valid:
            messagebox.showinfo("2048", "Game Over!!!")


board = Board()
game2048 = Game(board)
game2048.start()
