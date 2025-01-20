import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from Dataloader_all import SequenceParatopeDataset
from Models_new import CoreModelWithGNN
from torch_geometric.data import Data, Batch
import os
import numpy as np

# Configuration for model and training
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

config = {
    'batch_size': 8,
    'sequence_file': 'preprocessed_train_sequences.npz',
    'data_file': 'train_data.npz',
    'edge_file': 'train_edge_lists.npz',
    'max_len': 1200,
    'vocab_size': 22,
    'embed_dim': 256,
    'num_heads': 16,
    'num_layers': 1,
    'num_gnn_layers': 1,
    'num_classes': 2,
    'num_epochs': 1000,
    'learning_rate': 0.0001,
    'max_grad_norm': 0.1,
    'validation_split': 0.1,  # 10% for validation
    'early_stop_patience': 10,  # Stop if no improvement for 30 epochs
    'initial_gradient_noise_std': 0.05,  # Initial std for gradient noise
    'accumulation_steps': 2  # Gradient accumulation steps
}

def setup_ddp(rank, world_size):
    """
    Initialize the process group for Distributed Data Parallel (DDP).
    """
    init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """
    Destroy the process group after training.
    """
    destroy_process_group()

def add_gradient_noise(params, std_dev):
    """Add Gaussian noise to gradients."""
    for param in params:
        if param.grad is not None:
            noise = torch.normal(mean=0, std=std_dev, size=param.grad.shape, device=param.grad.device)
            param.grad.add_(noise)

def custom_collate_fn(batch):
    sequences, ss, sasa, pt, sapt, edge = zip(*batch)

    # Stack sequences into a single tensor
    sequence_tensor = torch.stack(sequences)  # Shape: [batch_size, 100, seq_len, feature_dim]

    # Convert ss, sasa, pt, and sapt to tensors
    ss_tensor = torch.tensor(np.array(ss), dtype=torch.long)
    sasa_tensor = torch.tensor(np.array(sasa), dtype=torch.float)
    pt_tensor = torch.tensor(np.array(pt), dtype=torch.long)
    sapt_tensor = torch.tensor(np.array(sapt), dtype=torch.float)

    # Create PyTorch Geometric Data objects for each graph
    graphs = []
    num_nodes = config['max_len']
    for i in range(len(edge)):
        dummy_features = torch.ones((num_nodes, 1))  # Dummy node features of size [num_nodes, 1]
        edge_index = edge[i].T.long()
        graph = Data(x=dummy_features, edge_index=edge_index)
        graphs.append(graph)

    # Batch graphs using PyTorch Geometric's Batch
    batched_graph = Batch.from_data_list(graphs)

    return batched_graph, sequence_tensor, ss_tensor, sasa_tensor, pt_tensor, sapt_tensor

def main(rank, world_size):
    setup_ddp(rank, world_size)

    # Dataset preparation
    dataset = SequenceParatopeDataset(data_file=config['data_file'], sequence_file=config['sequence_file'], edge_file=config['edge_file'], max_len=config['max_len'])
    val_size = int(len(dataset) * config['validation_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Distributed Sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=custom_collate_fn, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], collate_fn=custom_collate_fn, sampler=val_sampler)

    # Model setup
    device = torch.device(f'cuda:{rank}')
    model = CoreModelWithGNN(
        vocab_size=config['vocab_size'],
        seq_len=config['max_len'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        num_gnn_layers=config['num_gnn_layers'],
        num_classes=config['num_classes']
    ).to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training and validation loop
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        current_gradient_noise_std = config['initial_gradient_noise_std'] * (0.9 ** epoch)

        train_sampler.set_epoch(epoch)  # Shuffle the sampler for each epoch

        for i, (batched_graph, sequence_tensor, ss_tensor, sasa_tensor, pt_tensor, sapt_tensor) in enumerate(train_loader):
            if i % config['accumulation_steps'] == 0:
                optimizer.zero_grad(set_to_none=True)

            batched_graph = batched_graph.to(device)
            sequence_tensor, pt_tensor = sequence_tensor.to(device), pt_tensor.to(device)

            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                output_seq = model(sequence_tensor, batched_graph)
                output_seq = output_seq.view(-1, config['num_classes'])
                pt_tensor = pt_tensor.view(-1)
                loss = criterion(output_seq, pt_tensor) / config['accumulation_steps']

            scaler.scale(loss).backward()
            if current_gradient_noise_std > 0:
                add_gradient_noise(model.parameters(), current_gradient_noise_std)

            if (i + 1) % config['accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * config['accumulation_steps']

        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        print(f"[Rank {rank}] Epoch [{epoch + 1}/{config['num_epochs']}], Training Loss: {avg_loss:.4f}")

    cleanup_ddp()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)

