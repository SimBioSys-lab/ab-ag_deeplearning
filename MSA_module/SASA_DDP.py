import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from Dataloader_SS_SASA import SequenceSASADataset
from Models import SASAModel
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os
os.environ['NCCL_DEBUG'] = 'INFO'

# Configuration for model and training
config = {
    'batch_size': 16,
    'sequence_file': 'preprocessed_seq_ab_1200.npz',
    'sasa_file': 'sasa_data.csv',
    'seq_len': 1200,
    'vocab_size': 22,
    'embed_dim': 128,
    'num_heads': 8,
    'num_layers': 1,
    'num_epochs': 10,
    'learning_rate': 0.01
}

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Training function
def train_ddp(rank, world_size, config):
    setup(rank, world_size)

    # Load dataset and DistributedSampler
    dataset = SequenceSASADataset(sequence_file=config['sequence_file'], sasa_file=config['sasa_file'], max_len=config['seq_len'])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler)

    # Initialize the model and move to device
    model = SASAModel(vocab_size=config['vocab_size'], seq_len=config['seq_len'], embed_dim=config['embed_dim'],
                      num_heads=config['num_heads'], num_layers=config['num_layers']).to(rank)
    model = DDP(model, device_ids=[rank])

    # Set up loss function, optimizer, and scaler for AMP
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scaler = GradScaler()

    # Training loop
    for epoch in range(config['num_epochs']):
        sampler.set_epoch(epoch)  # Shuffle data differently at each epoch
        model.train()
        total_loss = 0

        for seq, tgt in dataloader:
            optimizer.zero_grad()

            # Move data to the correct device
            seq, tgt = seq.to(rank), tgt.to(rank)

            # Forward pass with autocast for mixed precision
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                output_seq = model(seq)
                mask = tgt < 0  # Assuming -1 is the padding value for targets
                tgt_padded = torch.where(mask, tgt, torch.zeros_like(tgt)).float()
                loss = (criterion(output_seq, tgt_padded) * mask.float()).mean()
                total_loss += loss.item()

            # Backward pass and optimization with AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        avg_loss = total_loss / len(dataloader)
        if rank == 0:  # Print from rank 0 to avoid duplicate logging
            print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Loss: {avg_loss:.4f}', flush=True)

    # Save model's specific layers from rank 0
    if rank == 0:
        pretrained_layers = {
            'SelfAttention': model.module.self_attention.state_dict()
        }
        torch.save(pretrained_layers, 'pretrained_SelfAttention_SASA.pth')
        print("Model layers saved successfully on rank 0.", flush=True)

    cleanup()

# Main entry for multi-processing
def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size, config), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

