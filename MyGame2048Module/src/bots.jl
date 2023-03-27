using MyGame2048: make_move, random_move, Game, init_game, MAX_VALUE

const SPM_SCALE_PARAM = 10
const SL_SCALE_PARAM = 12
const SEARCH_PARAM = 200
const NUMBER_OF_MOVES = 4
const MAX_MOVES = 5000

struct HeuristicBot
    game::Game

end

function HeuristicBot()
    HeuristicBot(init_game())
end

function get_search_params(number::Int64)
    searches_per_move = SPM_SCALE_PARAM * (1 + (number // SEARCH_PARAM))
    search_length = SL_SCALE_PARAM * (1 + (number // SEARCH_PARAM))
    searches_per_move, search_length
end
moves = []
function play(bot::HeuristicBot, move_number::Int64)
    possible_moves = ["l", "r", "u", "d"]
    move_scores = zeros(NUMBER_OF_MOVES)
    searches_per_move, search_length = get_search_params(move_number)
    old_game = deepcopy(bot.game)
    for index in 1:NUMBER_OF_MOVES
        move = possible_moves[index]
        game, is_valid = make_move(deepcopy(old_game), move)

        move_scores[index] += game.reward
        if !is_valid
            continue
        end

        for _ in 1:searches_per_move
            move_number = 1
            search_board = deepcopy(game.board)
            new_game = Game(search_board, game.reward)
            while move_number < search_length
                tmp_game, is_valid = random_move(new_game)
                if is_valid
                    new_game = tmp_game
                    move_scores[index] += tmp_game.reward
                end
                move_number += 1

            end
        end
    end
    best_move_index = argmax(move_scores)
    best_move = possible_moves[best_move_index]

    if rand() < 0.99
        best_game, is_valid = make_move(old_game, best_move)
    else
        best_game, is_valid = random_move(old_game)
    end
    best_game, is_valid
end

function play(bot::HeuristicBot)
    move_number = 0
    while true
        move_number += 1
        game, is_valid = play(bot, move_number)
        bot = HeuristicBot(game)
        if !is_valid
            break
        end
        if MAX_VALUE in bot.game.board
            break
        end
        if move_number == MAX_MOVES
            break
        end
    end
    println("Move number: ", move_number)
    bot.game
end
for i in 1:5
    bot = HeuristicBot()
    @time println(max(play(bot).board...))
end

