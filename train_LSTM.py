import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
from generate_data import generate_random_board
from score_data import score_board
import torch.distributions as dist


# Load dataset
df = pd.read_csv("boards.csv")
boards = df["board"].values

# Define letter encoding
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_to_index = {c: i for i, c in enumerate(LETTERS)}
index_to_char = {i: c for c, i in char_to_index.items()}

# Encode boards as sequences of integers
def encode_board(board):
    return [char_to_index[c] for c in board]

X_train = np.array([encode_board(board) for board in boards])
X_train = torch.tensor(X_train, dtype=torch.long)

# Define LSTM Model
class BoggleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BoggleLSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, 8)
        self.lstm = nn.LSTM(8, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)  # Get the last timestep output
        return x

# Model parameters
input_size = len(LETTERS)
hidden_size = 64
output_size = len(LETTERS) #prediect a letter at each time step

model = BoggleLSTM(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 16
batch_size = 32
num_batches = 100  # Define number of batches per epoch


def get_random_batch(batch_size):
    batch_boards = []
    for _ in range(batch_size):
        # Randomly pick one board from the dataset
        board_str = generate_random_board()  # Generate board from generate_data.py
        board_indices = [char_to_index[c] for c in board_str]
        batch_boards.append(board_indices)
    return torch.tensor(batch_boards, dtype=torch.long)  # (batch_size, 16)


if __name__ == "__main__":
    print("Training with REINFORCE loss using board scores as rewards...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for _ in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = get_random_batch(batch_size)  # (batch_size, 16)
            optimizer.zero_grad()
            
            outputs = model(batch)  # Shape: (batch_size, 16, 26)
            # Create a categorical distribution for each position based on logits (no softmax needed)
            m = dist.Categorical(logits=outputs)
            # Sample a board: shape (batch_size, 16)
            sampled_indices = m.sample()
            # Calculate log probabilities: shape (batch_size, 16)
            log_probs = m.log_prob(sampled_indices)
            # Sum log probabilities over the sequence: shape (batch_size,)
            total_log_prob = log_probs.sum(dim=1)
            
            # For each board in the batch, convert to a string and compute its reward
            rewards = []
            for seq in sampled_indices:
                board_str = ''.join(index_to_char[idx.item()] for idx in seq)
                # Compute the board score (reward)
                reward = score_board(board_str)
                rewards.append(reward)
            rewards = torch.tensor(rewards, dtype=torch.float)
            
            # Optionally: subtract a baseline to reduce variance (here, the batch mean)
            baseline = rewards.mean()
            advantages = rewards - baseline
            
            # REINFORCE loss: negative of (advantage * total log probability)
            loss = - (advantages * total_log_prob).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Save the model weights
    torch.save(model.state_dict(), "board_lstm.pth")
    print("Training complete. Model saved as 'board_lstm.pth'.")
