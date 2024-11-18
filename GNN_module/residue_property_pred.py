import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training function with autocast and GradScaler for mixed precision
def train_gat(model, dataloader, num_epochs=10, learning_rate=0.01):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # Loss for classification

    print(f"Total number of trainable parameters: {count_parameters(model)}", flush=True)

    # Initialize the GradScaler for mixed precision
    scaler = GradScaler('cuda')

    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in dataloader:
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass with mixed precision
            with autocast('cuda'):
                output = model(batch)  # Model forward pass with autocast
                labels = batch.y  # Assuming the labels (e.g., class labels) are stored in batch.y

                # Compute the loss
                loss = criterion(output, labels)
                total_loss += loss.item()

            # Backward pass and optimization with GradScaler
            scaler.scale(loss).backward()  # Scaled backward pass
            scaler.step(optimizer)  # Update the weights
            scaler.update()  # Update the scaler for the next iteration

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}', flush=True)

    print("Training completed!", flush=True)

