module MyGame2048

export init_game, random_move, make_move, Game, MAX_VALUE, HeuristicBot, play, SPM_SCALE_PARAM, SL_SCALE_PARAM, SEARCH_PARAM, NUMBER_OF_MOVES, MAX_MOVES

include("game.jl")
include("bots.jl")

end # module
