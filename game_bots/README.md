# 2048 game wrapper for Julia code in Python and bots
In this directory are located Python scripts which contain implementation of Graphical User Interface using Tkinter library, wrappers for Julia implementation of 2048 game, and three different types of bots that can play the game.

## Bots
The code that implements each bot is in three Python scripts in the directory game_bots/bots. There are therefore three types of bots:
- bot which learns to play a game using Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
- bot which learns to play a game using Deep Q-Learning or another machine learning bot that use Temporal Difference Learning (TD)
- bot which learns to play a game using some heuristic based on depth-first search
If you want to run selected bot please search specific function to run the agent. For example for heuristic bot this function is `run_py_heuristic_bot`.
In `weights` directory are stored weights for bots.
## GUI
GUI for 2048 game is located in `game_bots/gui/main.py`. To play the game with GUI please run this `main.py` script, use the arrows on the keyboard and enjoy!