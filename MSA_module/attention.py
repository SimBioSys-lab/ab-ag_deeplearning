import torch
import torch.nn as nn
import numpy as np

# Define the vocabulary (amino acids + padding and gap tokens)
VOCAB = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "-", "PAD"]
VOCAB_SIZE = len(VOCAB)

# Mapping amino acids to indices
token_to_idx = {token: idx for idx, token in enumerate(VOCAB)}
idx_to_token = {idx: token for token, idx in token_to_idx.items()}

# Function to read A3M file and convert sequences to token indices
def read_a3m_file(a3m_path):
    """
    Reads an A3M file and converts sequences to token indices.
    """
    sequences = []
    with open(a3m_path, 'r') as f:
        for line in f:
            if not line.startswith(">"):
                # Convert the sequence to a list of indices
                tokenized_seq = [token_to_idx[res] for res in line.strip() if res in VOCAB]
                sequences.append(tokenized_seq)
    
    # Pad sequences to the same length (necessary for batching)
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = [seq + [token_to_idx['PAD']] * (max_len - len(seq)) for seq in sequences]
    return padded_sequences

# Positional encoding module
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Main model that uses the trainable tokenizer, positional encoding, and attention-based mechanism
class SequenceInteractionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super(SequenceInteractionModel, self).__init__()
        
        # Trainable embedding layer for tokenization
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=token_to_idx["PAD"])
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final prediction layer (e.g., for structural prediction)
        self.fc = nn.Linear(embed_dim, 1)  # Output size depends on prediction task (e.g., contact map, binding score)

    def forward(self, sequences):
        # Token embedding (turn sequences into embeddings)
        embedded = self.embedding(sequences)
        
        # Add positional encoding
        embedded = self.positional_encoding(embedded)
        
        # Pass through Transformer encoder to capture interaction between tokens
        encoded_sequences = self.transformer_encoder(embedded)
        
        # Aggregate outputs (e.g., take mean across sequence)
        output = encoded_sequences.mean(dim=1)
        
        # Final prediction
        prediction = self.fc(output)
        return prediction

# Example usage: reading an A3M file, tokenizing, and passing through the model
a3m_file = "t000_upper.a3m"
sequences = read_a3m_file(a3m_file)
sequences_tensor = torch.tensor(sequences)  # Convert to tensor for model input

# Instantiate and run the model
model = SequenceInteractionModel(vocab_size=VOCAB_SIZE, embed_dim=64, num_heads=4, num_layers=2)

# Forward pass with tokenized sequences
predictions = model(sequences_tensor)
print(predictions.shape)  # Shape will depend on prediction task (e.g., [batch_size, 1])

# Optional: Training the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()  # Depends on the task (MSE for regression, CrossEntropy for classification)

# Example dummy labels for training
target_labels = torch.randn(predictions.shape)  # Replace with actual labels

# Example training loop
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(sequences_tensor)
    loss = loss_fn(predictions, target_labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

