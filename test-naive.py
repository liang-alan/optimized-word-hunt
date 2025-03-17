import generate_data  
import score_data
from tqdm import tqdm 


num_iterations = 1
total_score = 0

for _ in tqdm(range(num_iterations), desc="Processing Boards"):
    board = generate_data.generate_random_board() 
    score = score_data.score_board(board) 
    total_score += score
    # print(f"Board: {board} | Score: {score}") 

# Compute and print the average score
average_score = total_score / num_iterations
print(f"\nAverage Score over {num_iterations} iterations: {average_score}")