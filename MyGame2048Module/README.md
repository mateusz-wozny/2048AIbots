# 2048 game in Julia
In this directory is located code for 2048 game and implementation of heuristic bot in Julia. 

## Game
The whole code of game is in `src/game.jl` file. If you want to play one game please type `play_game` function e.g. in the bottom of game file, run script, write in terminal letter corresponding to move (that means left->l, right->r, up->u, down->d) and enjoy the game!

## Heuristic bot
In `src/bots.jl` is located code which implements heuristic bot in Julia. This is made for speed comparison  between Julia and Python speed.
If you want to run the bot please write below code in the bottom of bots file
```julia
bot = HeuristicBot()
game = play(bot)
game.board, game.reward
```
This code will return move number made by bot, game board and game score.