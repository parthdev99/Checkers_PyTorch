import numpy as np

def best_move(board):
    compressed_board = compress(board)
    boards = np.zeros((0, 32))
    boards = generate_next(board)

    # Convert data to torch tensor
    boards = torch.tensor(boards, dtype=torch.float32)

    # Forward pass through the reinforcement model
    outputs = reinforced_model(boards)
    scores = outputs.detach().numpy()

    max_index = np.argmax(scores)
    best = boards[max_index]
    return best

def print_board(board):
  for row in board:
    for square in row:
      if square == 1:
        caracter = "|O"
      elif square == -1:
        caracter = "|X"
      else:
        caracter = "| "
      print(str(caracter), end='')
    print('|')


start_board = [1, 1, 1, 1,  1, 1, 1, 0,  1, 0, 0, 1,  0, 1, 1, 0,  0, 0, 0, 0,  0, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1]
start_board = expand(start_board)
next_board = expand(best_move(start_board))

print("Starting position : ")
print_board(start_board)

print("\nBest next move : ")
print_board(next_board)