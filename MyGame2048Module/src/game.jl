using StatsBase
# # Define the game board as a 4x4 matrix of zeros
const MAX_VALUE = 8192

mutable struct Game
    board::Matrix{Int}
    reward::Int
end

Base.display(game::Game) = display(game.board)
Base.print(game::Game) = print(game.board)
Base.println(game::Game) = println(game.board)

function init_game()::Game
    zeros(Int, 4, 4) |> x -> Game(x, 0) |> add_tile
end

# Function to print the game board
function print_board(board::Matrix{Int})
    display(board)
end

# Function to add a new random tile (either 2 or 4) to the board
function add_tile(game::Game)
    # Find all empty positions on the board
    empty_positions = findall(x -> x == 0, game.board)
    # Choose a random empty position
    pos = empty_positions[rand(1:length(empty_positions))]
    # Choose a random value (either 2 or 4)
    value = rand() < 0.9 ? 2 : 4
    # Set the value at the chosen position
    game.board[pos] = value
    game
end

# Function to shift tiles to the left
function shift_left(game::Game)
    update = false
    old_board = deepcopy(game.board)
    for i in 1:size(game.board, 1)
        # Remove zeros
        row = filter(x -> x != 0, game.board[i, :])
        # Merge adjacent tiles with the same value
        for j in 1:length(row)-1
            if row[j] == row[j+1]
                row[j] *= 2
                row[j+1] = 0
                game.reward += row[j]
                update = true
            end
        end

        # Add zeros back to the end of the row
        row = vcat(row, zeros(Int, size(game.board, 2) - length(row)))
        # Update the board
        game.board[i, :] = row

    end
    game, sum(old_board .== game.board) / 16 != 1 || sum(game.board .== 0) > 0
end

# Function to shift tiles to the right
function shift_right(game::Game)
    # Reverse the board, shift left, and reverse back
    reverse!(game.board, dims=2)
    game, is_valid = shift_left(game)
    reverse!(game.board, dims=2)
    game, is_valid
end

# Function to shift tiles up
function shift_up(game::Game)
    # Transpose the board, shift left, and transpose back
    game.board = transpose(game.board)
    game, is_valid = shift_left(game)
    game.board = transpose(game.board)
    game, is_valid
end

# Function to shift tiles down
function shift_down(game::Game)
    # Transpose the board, shift right, and transpose back
    game.board = transpose(game.board)
    game, is_valid = shift_right(game)
    game.board = transpose(game.board)
    game, is_valid
end

# Start the game loop
function play_game()
    game = init_game()
    while true
        # Add a new tile to the board
        # Print the board
        display(game)
        # Check if the game is over (no more empty spaces or no more valid moves)
        if sum(game.board .== 0) == 0
            println("Game over!")
            break
        end
        # Ask the user for a move
        println("Enter a move l,r,u,d:")
        move = Symbol(readline())
        # Apply the move
        if move == :l
            shift_left(game) |> add_tile
        elseif move == :r
            shift_right(game) |> add_tile
        elseif move == :u
            shift_up(game) |> add_tile
        elseif move == :d
            shift_down(game) |> add_tile
        else
            println("Invalid move!")
        end


    end
end
function random_move(game::Game)
    move = sample(["l", "r", "u", "d"], Weights([0.4, 0.15, 0.05, 0.4]))
    make_move(game, move)
end
function make_move(game::Game, move::String)
    move = Symbol(move)
    # Apply the move
    if move == :l
        game, is_valid = shift_left(game)
        if is_valid
            add_tile(game)
        end
    elseif move == :r
        game, is_valid = shift_right(game)
        if is_valid
            add_tile(game)
        end
    elseif move == :u
        game, is_valid = shift_up(game)
        if is_valid
            add_tile(game)
        end
    elseif move == :d
        game, is_valid = shift_down(game)
        if is_valid
            add_tile(game)
        end
    else
        println("Invalid move!")
    end
    game, is_valid
end

function vector(game::Game)
    vec = zeros(Int8, 208)
    for (i, num) in enumerate(vcat(game.board...))
        if num > 0
            num = Int(log2(num))
        end
        vec[num*16+i] = 1
    end
    vec
end
function is_move_available(game::Game, move::String)
    _, is_valid = make_move(game, move)
    return is_valid
end
function game_over(game::Game)
    for move in ["l", "r", "u", "d"]
        if is_move_available(game, move)
            return false
        end
    end
    return true
end
game = init_game()
println(game_over(game))