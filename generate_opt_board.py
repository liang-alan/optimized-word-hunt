import torch
import numpy as np
from train_LSTM import BoggleLSTM  # Make sure your BoggleLSTM is accessible from this import
from generate_data import generate_random_board
from score_data import score_board  # Your board scoring function

# Define our letters and create mappings for encoding/decoding
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_to_index = {c: i for i, c in enumerate(LETTERS)}
index_to_char = {i: c for i, c in enumerate(LETTERS)}

# Model parameters must match what was used during training
input_size = len(LETTERS)   # 26 letters
hidden_size = 64
num_classes = len(LETTERS)  # 26 classes

# Instantiate the model and load the trained weights
model = BoggleLSTM(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load("board_lstm.pth"))
model.eval()

def generate_optimized_board(num_candidates=10, temperature=0.8):
    """
    Generate several candidate boards using sampling with temperature,
    then return the seed board, a list of candidate boards with their scores,
    and the candidate board with the highest score.
    
    Args:
      num_candidates (int): Number of candidate boards to sample.
      temperature (float): Temperature to adjust the softness of the distribution.
      
    Returns:
      seed_board (str): The original seed board.
      candidate_boards (list of tuples): Each tuple contains (board_str, board_score).
      best_board (str): The candidate board with the highest score.
      best_score (float): The best candidate's score.
    """
    best_board = None
    best_score = -float("inf")
    candidate_boards = []
    
    # Generate a random board as a seed
    seed_board = generate_random_board()
    seed = torch.tensor([char_to_index[c] for c in seed_board], dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(seed)  # Shape: (1, 16, 26)
    
    # Adjust the logits with temperature: lower temperature makes distribution sharper.
    outputs = outputs / temperature
    probabilities = torch.softmax(outputs, dim=-1)  # Shape: (1, 16, 26)
    
    for _ in range(num_candidates):
        # Sample letters for each of the 16 positions.
        sampled_indices = torch.multinomial(probabilities.view(-1, num_classes), num_samples=1)
        sampled_indices = sampled_indices.view(1, -1)  # Shape: (1, 16)
        
        # Convert sampled indices to board string
        candidate_board = ''.join(index_to_char[idx.item()] for idx in sampled_indices[0])
        candidate_score = score_board(candidate_board)
        candidate_boards.append((candidate_board, candidate_score))
        
        if candidate_score > best_score:
            best_score = candidate_score
            best_board = candidate_board
            
    return seed_board, candidate_boards, best_board, best_score

if __name__ == "__main__":
    seed_board, candidate_boards, best_board, best_score = generate_optimized_board(num_candidates=10, temperature=0.8)
    
    print("Original Seed Board:", seed_board)
    print("Seed Board Score:", score_board(seed_board))
    print("\nCandidate Boards:")
    for board, score in candidate_boards:
        print(f"Board: {board}  Score: {score}")
        
    print("\nBest Candidate Board:", best_board)
    print("Best Score:", best_score)
