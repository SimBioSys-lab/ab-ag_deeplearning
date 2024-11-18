import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import einsum, nn
from einops import rearrange, repeat
import math
from torch.amp import autocast
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix
import numpy as np
from scipy.sparse import csr_matrix

# Helpers
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)

# Attention
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, gating=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.0)
        nn.init.constant_(self.gating.bias, 1.0)
        self.dropout = nn.Dropout(dropout)
        init_zero_(self.to_out)
    def forward(self, x, mask=None, tied=False):
        h = self.heads
        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        # Scale
        q = q * self.scale
        # Attention
        dots = einsum("b h i d, b h j d -> b h i j", q, k)
        # Tied attention in row dimension
        if tied:
            # Calculate the row-wise (h-dimension) average
            rowwise_average = torch.mean(dots, dim=3, keepdim=True)  # Averaging over the 'j' dimension since this dimension is going to summed over

            # Calculate the square root of b (batch size)
            b_sqrt = math.sqrt(dots.size(0))  # Assuming b is the batch size dimension

            # Divide the result by the square root of b
            dots = rowwise_average / b_sqrt

            # Broadcast the result back to shape (b, h, i, j) for the next einsum step
            dots = rowwise_average.expand_as(dots)

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # Aggregate
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        # Gating
        gates = self.gating(x)
        out = out * gates.sigmoid()

        # Combine to output
        return self.to_out(out) 

class AxialAttention(nn.Module):
    def __init__(self, dim, heads, row_attn=True, col_attn=True, **kwargs):
        super().__init__()
        assert row_attn or col_attn, "Either row or column attention must be turned on."

        self.row_attn = row_attn
        self.col_attn = col_attn
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, heads=heads, **kwargs)


    def forward(self, x, edges=None, mask=None):
        assert self.row_attn ^ self.col_attn, "Has to be either row or column attention, not both."

        b, h, w, d = x.shape
        x = self.norm(x)

        # Axial attention setup
        if self.col_attn:
            axial_dim = w
            input_fold_eq = "b h w d -> (b w) h d"
            output_fold_eq = "(b w) h d -> b h w d"
            tied = False
        elif self.row_attn:
            axial_dim = h
            input_fold_eq = "b h w d -> (b h) w d"
            output_fold_eq = "(b h) w d -> b h w d"
            tied = True
        x = rearrange(x, input_fold_eq)

        out = self.attn(x, mask=mask,  tied = tied)
        return rearrange(out, output_fold_eq, h=h, w=w)


class MSASelfAttentionBlock(nn.Module):
    def __init__(self, dim, seq_len, heads, dim_head, dropout=0.0):
        super().__init__()
        self.row_attn = AxialAttention(dim=dim, heads=heads, row_attn=True, col_attn=False)
        self.col_attn = AxialAttention(dim=dim, heads=heads, row_attn=False, col_attn=True)

    def forward(self, x, mask=None, pairwise_repr=None):
        x = self.row_attn(x, mask=mask, edges=pairwise_repr) + x
        x = self.col_attn(x, mask=mask) + x
        return x



class BasicModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers):
        super().__init__()
        print(f"Initializing BasicModel with {num_layers} repeated layers...")

        self.seq_len = seq_len
        self.num_layers = num_layers  # Number of times to apply the layers

        # Separate Embedding layers for sequences1 and sequences2
        self.embedding1 = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding2 = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Separate Self-Attention layers for sequences1 and sequences2 (used repeatedly)
        self.self_attention1 = MSASelfAttentionBlock(dim=embed_dim, seq_len=seq_len, heads=num_heads, dim_head=64, dropout=0.3)
        self.self_attention2 = MSASelfAttentionBlock(dim=embed_dim, seq_len=seq_len, heads=num_heads, dim_head=64, dropout=0.3)

        # Shared Cross-Attention layer (used repeatedly)
        self.cross_attention = MSABidirectionalCrossAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, sequences1, sequences2):
        # Ensure input tensors are on the same device as the model
        sequences1 = sequences1.to(next(self.parameters()).device)
        sequences2 = sequences2.to(next(self.parameters()).device)

        with autocast('cuda'):
            # Separate embedding for sequences1 and sequences2
            output1 = self.embedding1(sequences1)
            output2 = self.embedding2(sequences2)

            # Reuse the same self-attention and cross-attention layers for num_layers times
            for _ in range(self.num_layers):
                # Apply separate self-attention to each sequence
                output1 = self.self_attention1(output1)
                output2 = self.self_attention2(output2)

                # Apply shared cross-attention between the two sequences
                output1, output2 = self.cross_attention(output1, output2)

        return output1, output2

class MSAModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers):
        super().__init__()
        print("Initializing MSAModel...")

        # Use BasicModel for embedding and attention
        self.basic_model = BasicModel(vocab_size, seq_len, embed_dim, num_heads, num_layers)

        # Fully connected layer and sigmoid for predictions
        self.fc = nn.Linear(embed_dim * seq_len, seq_len)  # Fully connected layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, sequences1, sequences2):
        # Ensure input tensors are on the same device as the model
        sequences1 = sequences1.to(next(self.parameters()).device)
        sequences2 = sequences2.to(next(self.parameters()).device)

        with autocast('cuda'):
            # Pass the input sequences through the basic model
            output1, output2 = self.basic_model(sequences1, sequences2)

            # Reshape the output to match the input for the fully connected layer
            batch_size = output1.shape[0]
            output1 = output1[:, 0, :, :].reshape(batch_size, -1)
            output2 = output2[:, 0, :, :].reshape(batch_size, -1)

            # Apply fully connected layers to generate predictions
            prediction1 = self.fc(output1)
            prediction2 = self.fc(output2)

    def predict(self, sequences1, sequences2):
        # Get the logits from the forward pass
        logits1, logits2 = self.forward(sequences1, sequences2)

        # Apply sigmoid to get probabilities
        probs1 = self.sigmoid(logits1)
        probs2 = self.sigmoid(logits2)

        # Convert to boolean using threshold of 0.5
        bool_output1 = probs1 >= 0.5
        bool_output2 = probs2 >= 0.5

        return bool_output1, bool_output2


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from scipy.sparse import csr_matrix
import numpy as np

class CheatSheetModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers, in_channels, hidden_channels, out_channels, threshold=0.5):
        super().__init__()
        print("Initializing MSAModel...")

        self.num_layers = num_layers
        self.threshold = threshold  # Threshold for creating new edges based on learned features

        # Use BasicModel for embedding and attention
        self.basic_model = BasicModel(vocab_size, seq_len, embed_dim, num_heads, num_layers)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=0.6)
        self.conv2 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=0.6)
        # Shared Cross-Attention layer (used repeatedly)
        self.cross_attention = MSABidirectionalCrossAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, sequences1, sequences2, edge_index1, edge_index2):
        # Ensure input tensors are on the same device as the model
        sequences1 = sequences1.to(next(self.parameters()).device)
        sequences2 = sequences2.to(next(self.parameters()).device)

        with torch.cuda.amp.autocast(enabled=True):
            # Pass the input sequences through the basic model
            output1, output2 = self.basic_model(sequences1, sequences2)

            batch_size = output1.shape[0]
            output1 = output1[:, 0, :, :].reshape(batch_size, -1)
            output2 = output2[:, 0, :, :].reshape(batch_size, -1)

            for _ in range(self.num_layers):
                # First GAT layer + activation
                output1 = self.conv1(output1, edge_index1)
                output1 = F.elu(output1)
                output2 = self.conv2(output2, edge_index2)
                output2 = F.elu(output2)

                # Apply shared cross-attention between the two sequences
                output1, output2 = self.cross_attention(output1, output2)

                # Dynamically update the adjacency matrix based on learned features
                attention_weights_1 = self.cross_attention.attention_weights_1
                attention_weights_2 = self.cross_attention.attention_weights_2
                updated_adj_matrix1 = self.update_adjacency_matrix(attention_weights_1)
                updated_adj_matrix2 = self.update_adjacency_matrix(attention_weights_2)

                # Recompute the edge index from the updated adjacency matrices
                edge_index1, _ = from_scipy_sparse_matrix(updated_adj_matrix1)
                edge_index2, _ = from_scipy_sparse_matrix(updated_adj_matrix2)

        return output1, output2

    def update_adjacency_matrix(self, attention_scores):
        """Update adjacency matrix based on the attention scores and a defined threshold."""
        # Average over heads to get a single attention score matrix for each pair
        avg_attention_scores = attention_scores.mean(dim=1)  # Assuming shape [batch, heads, seq_len, seq_len]

        # Convert attention scores to an adjacency matrix
        batch_size, seq_len, _ = avg_attention_scores.shape
        adjacency_matrices = []

        for i in range(batch_size):
            attention_matrix = avg_attention_scores[i]
            adj_matrix = (attention_matrix > self.threshold).float().cpu().numpy()
            adjacency_matrices.append(csr_matrix(adj_matrix))

        # Return the adjacency matrix of the first example (can be adapted for the whole batch)
        return adjacency_matrices[0] if batch_size == 1 else csr_matrix(np.mean(adjacency_matrices, axis=0))

class MSABidirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by the number of heads."

        # Linear layers for Q, K, V transformations
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # Output linear transformations
        self.out1 = nn.Linear(embed_dim, embed_dim)
        self.out2 = nn.Linear(embed_dim, embed_dim)

        # Scaling factor for dot-product attention
        self.scale = self.head_dim ** 0.5

    def forward(self, mat1, mat2):
        batch_size, num_sequences, seq_len, embed_dim = mat1.shape

        # mat1 attends to mat2
        Q1 = self.q_linear(mat1)
        K2 = self.k_linear(mat2)
        V2 = self.v_linear(mat2)
        Q1 = rearrange(Q1, "b n l (h d) -> b n h l d", h=self.num_heads)
        K2 = rearrange(K2, "b n l (h d) -> b n h l d", h=self.num_heads)
        V2 = rearrange(V2, "b n l (h d) -> b n h l d", h=self.num_heads)

        attention_scores_1 = torch.matmul(Q1, K2.transpose(-2, -1)) / self.scale
        self.attention_weights_1 = F.softmax(attention_scores_1, dim=-1)
        mat1_to_mat2_attention = torch.matmul(self.attention_weights_1, V2)
        mat1_to_mat2_attention = rearrange(mat1_to_mat2_attention, "b n h l d -> b n l (h d)")
        mat1_to_mat2_attention = self.out1(mat1_to_mat2_attention)

        # mat2 attends to mat1
        Q2 = self.q_linear(mat2)
        K1 = self.k_linear(mat1)
        V1 = self.v_linear(mat1)
        Q2 = rearrange(Q2, "b n l (h d) -> b n h l d", h=self.num_heads)
        K1 = rearrange(K1, "b n l (h d) -> b n h l d", h=self.num_heads)
        V1 = rearrange(V1, "b n l (h d) -> b n h l d", h=self.num_heads)

        attention_scores_2 = torch.matmul(Q2, K1.transpose(-2, -1)) / self.scale
        self.attention_weights_2 = F.softmax(attention_scores_2, dim=-1)
        mat2_to_mat1_attention = torch.matmul(self.attention_weights_2, V1)
        mat2_to_mat1_attention = rearrange(mat2_to_mat1_attention, "b n h l d -> b n l (h d)")
        mat2_to_mat1_attention = self.out2(mat2_to_mat1_attention)

        return mat1_to_mat2_attention, mat2_to_mat1_attention

