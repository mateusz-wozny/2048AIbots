# 2048AIbots
This repository contains implementation of 2048 game written in Julia and 3 types of bots which can play the game. Bots are written in Python.

## Usage
First of all you have to install Python and Julia. Next follow steps describe in [PyJulia](https://pyjulia.readthedocs.io/en/latest/index.html). If you have Julia and PyJulia follow below steps:
```bash
julia
```
next type ] in terminal. This will open package manager for Julia.
```bash
]
```
Next you have to develop local package for 2048 game.
```julia
dev /path/to/MyGame2048Module
```
Now you can use Julia `MyGame2048Module` module in Python. For more informations follow `README.md` in `MyGame2048Module` directory (if you want to play a game in terminal) or in `game_bots` (if you want to play a game using GUI or run bots).