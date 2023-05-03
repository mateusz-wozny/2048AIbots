using LinearAlgebra
# # Define the game board as a 4x4 matrix of zeros
const MAX_VALUE = 8192
const MOVES_MAPPING = Dict("l" => 0, "u" => 1, "r" => 2, "d" => 3)
mutable struct Game
    board::Matrix{Int}
    reward::Int
end

Base.display(game::Game) = display(game.board)
Base.print(game::Game) = print(game.board)
Base.println(game::Game) = println(game.board)

function init_game(binary::Bool=false)::Game
    zeros(Int, 4, 4) |> x -> Game(x, 0) |> x -> add_tile(x, binary)
end

# Function to print the game board
function print_board(board::Matrix{Int})
    display(board)
end

# Function to add a new random tile (either 2 or 4) to the board
function add_tile(game::Game, binary::Bool=false)
    # Find all empty positions on the board
    empty_positions = findall(x -> x == 0, game.board)
    # Choose a random empty position
    pos = empty_positions[rand(1:length(empty_positions))]
    factor = binary ? 1 : 2
    # Choose a random value (either 2 or 4)
    value = rand() < 0.9 ? 1 * factor : 2 * factor
    # Set the value at the chosen position
    game.board[pos] = value
    game
end

function add_tile(tuple::Tuple{Game,Bool})
    add_tile(tuple[1])
end

# Function to shift tiles to the left
function shift_left(game::Game, binary::Bool=false)
    update = false
    old_board = deepcopy(game.board)
    for i in 1:size(game.board, 1)
        # Remove zeros
        row = filter(x -> x != 0, game.board[i, :])
        # Merge adjacent tiles with the same value
        for j in 1:length(row)-1
            if row[j] == row[j+1]
                row[j] = binary ? row[j] + 1 : row[j] * 2
                row[j+1] = 0
                game.reward += binary ? 2^row[j] : row[j]
                update = true
            end
        end
        row = filter(x -> x != 0, row)
        # Add zeros back to the end of the row
        row = vcat(row, zeros(Int, size(game.board, 2) - length(row)))
        # Update the board
        game.board[i, :] = row

    end
    game, sum(old_board .== game.board) / 16 != 1 || sum(game.board .== 0) > 0
end

# Function to shift tiles to the right
function shift_right(game::Game, binary::Bool=false)
    # Reverse the board, shift left, and reverse back
    reverse!(game.board, dims=2)
    game, is_valid = shift_left(game, binary)
    reverse!(game.board, dims=2)
    game, is_valid
end

# Function to shift tiles up
function shift_up(game::Game, binary::Bool=false)
    # Transpose the board, shift left, and transpose back
    game.board = transpose(game.board)
    game, is_valid = shift_left(game, binary)
    game.board = transpose(game.board)
    game, is_valid
end

# Function to shift tiles down
function shift_down(game::Game, binary::Bool=false)
    # Transpose the board, shift right, and transpose back
    game.board = transpose(game.board)
    game, is_valid = shift_right(game, binary)
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
    available_moves = []
    for move in ["l", "r", "u", "d"]
        if is_move_available(MOVES_MAPPING[move], game.board)
            push!(available_moves, move)
        end
    end
    if available_moves == []
        push!(available_moves, "l")
    end
    move = rand(available_moves)
    make_move(game, move)
end

function pre_move(game::Game, move::String, binary::Bool=false)
    move = Symbol(move)
    # Apply the move
    if move == :l
        game, is_valid = shift_left(game, binary)

    elseif move == :r
        game, is_valid = shift_right(game, binary)

    elseif move == :u
        game, is_valid = shift_up(game, binary)

    elseif move == :d
        game, is_valid = shift_down(game, binary)

    else
        println("Invalid move!")
    end
    game, is_valid
end

function make_move(game::Game, move::String, binary::Bool=false)
    move = Symbol(move)
    # Apply the move
    if move == :l
        game, is_valid = shift_left(game, binary)
        if is_valid
            add_tile(game, binary)
        end
    elseif move == :r
        game, is_valid = shift_right(game, binary)
        if is_valid
            add_tile(game, binary)
        end
    elseif move == :u
        game, is_valid = shift_up(game, binary)
        if is_valid
            add_tile(game, binary)
        end
    elseif move == :d
        game, is_valid = shift_down(game, binary)
        if is_valid
            add_tile(game, binary)
        end
    else
        println("Invalid move!")
    end
    game, is_valid
end

function vector(matrix::Matrix)
    vec = zeros(208)
    for (i, num) in enumerate(vcat(matrix...))
        if num > 0
            num = Int(log2(num))
        end
        vec[num*16+i] = 1
    end
    vec
end

function check_state(state::Matrix)
    for row in 1:4
        has_empty = false
        for col in 1:4
            has_empty |= state[row, col] == 0
            if state[row, col] != 0 && has_empty
                return true
            end
            if state[row, col] != 0 && col > 1 && state[row, col] == state[row, col-1]
                return true
            end
        end
    end
    return false
end

function is_move_available(move, state::Matrix)
    temp_state = rotl90(state, move)
    return check_state(temp_state)
end
