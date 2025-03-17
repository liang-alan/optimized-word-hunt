import generate_data 
import score_data 
import pandas as pd
from tqdm import tqdm

num_boards = 10000  # Total boards to generate
top_n = 5000  # Keep only the top-scoring boards

boards_with_scores = []

print("Generating and scoring boards...")
for _ in tqdm(range(num_boards), desc="Processing Boards"):
    board = generate_data.generate_random_board()
    score = score_data.score_board(board)
    boards_with_scores.append((board, score))

# Sort boards by highest score and keep top N
boards_with_scores.sort(key=lambda x: x[1], reverse=True)
top_boards = boards_with_scores[:top_n]

# Save dataset to CSV
df = pd.DataFrame(top_boards, columns=["board", "score"])
df.to_csv("boards.csv", index=False)
print(f"Saved {top_n} high-scoring boards to 'boards.csv'")