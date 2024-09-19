import torch
import torch.nn as nn
import numpy as np
from DataLoader_att import A3MDataset
from Models import MSAModel





a3m_file = "test_256.a3m"
dataset = A3MDataset(a3m_file)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model, loss function, and optimizer
model = MSAModel(vocab_size=21, max_seq_len=512, embed_dim=128, num_heads=8, num_layers=1)  # Assuming 20 amino acids + 1 unknown token
criterion = nn.MSELoss()  # Loss for structure prediction (can be different based on target)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)

        # Example target: random structure values (replace with actual target)
        targets = torch.randn(batch.size(0), 1)

        # Compute loss
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader)}')

print("Training finished!")
