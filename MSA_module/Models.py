import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import einsum, nn
from torch_geometric.nn import GCNConv
from einops import rearrange, repeat
import math
from torch.amp import autocast

# Helpers
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)

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

        # Initialize gating layer
        nn.init.constant_(self.gating.weight, 0.0)
        nn.init.constant_(self.gating.bias, 1.0)

        self.dropout = nn.Dropout(dropout)

        # LayerNorm for residual connection
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None, tied=False):
        h = self.heads

        # Layer normalization before self-attention

        # Generate query, key, and value
        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # Scale query
        q = q * self.scale

        # Compute attention scores
        dots = einsum("b h i d, b h j d -> b h i j", q, k)

        # Tied attention in row dimension if specified
        if tied:
            rowwise_average = torch.mean(dots, dim=3, keepdim=True)
            b_sqrt = math.sqrt(dots.size(0))
            dots = rowwise_average / b_sqrt
            dots = rowwise_average.expand_as(dots)

        # Apply softmax to get attention probabilities
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # Compute output from values
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        # Gating mechanism
        gates = self.gating(x)
        out = out * gates.sigmoid()

        # Final linear layer
        out = self.to_out(out)

        # Apply residual connection
        out = x + out
        out = self.norm(out)

        return out

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

        # Layer Normalization layers
        self.norm = nn.LayerNorm(embed_dim)

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
        attention_weights_1 = F.softmax(attention_scores_1, dim=-1)
        mat1_to_mat2_attention = torch.matmul(attention_weights_1, V2)
        mat1_to_mat2_attention = rearrange(mat1_to_mat2_attention, "b n h l d -> b n l (h d)")
        mat1_to_mat2_attention = self.out1(mat1_to_mat2_attention)

        # residual connection for mat1
        mat1_to_mat2_attention = self.norm(mat2 + mat1_to_mat2_attention)

        # mat2 attends to mat1
        Q2 = self.q_linear(mat2)
        K1 = self.k_linear(mat1)
        V1 = self.v_linear(mat1)
        Q2 = rearrange(Q2, "b n l (h d) -> b n h l d", h=self.num_heads)
        K1 = rearrange(K1, "b n l (h d) -> b n h l d", h=self.num_heads)
        V1 = rearrange(V1, "b n l (h d) -> b n h l d", h=self.num_heads)

        attention_scores_2 = torch.matmul(Q2, K1.transpose(-2, -1)) / self.scale
        attention_weights_2 = F.softmax(attention_scores_2, dim=-1)
        mat2_to_mat1_attention = torch.matmul(attention_weights_2, V1)
        mat2_to_mat1_attention = rearrange(mat2_to_mat1_attention, "b n h l d -> b n l (h d)")
        mat2_to_mat1_attention = self.out2(mat2_to_mat1_attention)

        # residual connection for mat2
        mat2_to_mat1_attention = self.norm(mat1 + mat2_to_mat1_attention)

        return mat2_to_mat1_attention, mat1_to_mat2_attention

class SASAModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers):
        super().__init__()
        print(f"Initializing SASAModel with {num_layers} self-attention layers...")

        self.seq_len = seq_len
        self.num_layers = num_layers

        # Embedding layer for sequences
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multiple self-attention blocks, each with its own parameters
        self.self_attention_layers = nn.ModuleList([
            MSASelfAttentionBlock(dim=embed_dim, seq_len=seq_len, heads=num_heads, dim_head=64, dropout=0.3)
            for _ in range(num_layers)
        ])

        # Fully connected layers for SASA prediction
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.fc2 = nn.Linear(embed_dim // 2, 1)  # Output one value per sequence position

    def forward(self, sequences):
        # Assume input tensors are on the correct device

        # Get the embeddings for the input sequences
        output = self.embedding(sequences)

        # Apply each self-attention layer sequentially
        for attention_layer in self.self_attention_layers:
            output = attention_layer(output)
        
        # Reshape the output for the fully connected layers
        batch_size, num_sequences, seq_len, embed_dim = output.shape
        output = output[:, 0, :, :]  # Shape: (batch_size, seq_len, embed_dim)

        # Pass through the fully connected layers
        output = self.fc1(output)
        output = torch.relu(output)
        output = self.fc2(output).squeeze(-1)  # Shape: (batch_size, seq_len)

        return output

class SecondaryStructureModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers, num_classes=8):
        super().__init__()
        print(f"Initializing SecondaryStructureModel with {num_layers} self-attention layers...")

        self.seq_len = seq_len
        self.num_classes = num_classes

        # Embedding layer for sequences
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multiple self-attention blocks, each with its own parameters
        self.self_attention_layers = nn.ModuleList([
            MSASelfAttentionBlock(dim=embed_dim, seq_len=seq_len, heads=num_heads, dim_head=64, dropout=0.3)
            for _ in range(num_layers)
        ])

        # Fully connected layer for classification
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, sequences):
        # Embed the sequences
        output = self.embedding(sequences)

        # Apply each self-attention layer sequentially
        for attention_layer in self.self_attention_layers:
            output = attention_layer(output)

        # Reshape the output for the fully connected layer
        batch_size, num_sequences, seq_len, embed_dim = output.shape
        output = output[:, 0, :, :]

        # Apply the fully connected layer to get class scores for each position
        output = self.fc(output)
        return output  # Shape: [batch_size, seq_len, num_classes]

class ParatopeModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers, num_classes=2):
        super().__init__()
        print(f"Initializing ParatopeModel with {num_layers} self-attention layers...")

        self.seq_len = seq_len
        self.num_classes = num_classes

        # Embedding layer for sequences (kept on CPU)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multiple self-attention blocks, each with its own parameters (on GPU by default)
        self.self_attention_layers = nn.ModuleList([
            MSASelfAttentionBlock(dim=embed_dim, seq_len=seq_len, heads=num_heads, dim_head=64, dropout=0.3)
            for _ in range(num_layers)
        ])

        # Fully connected layer for classification (on GPU by default)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, sequences):
        # Embed the sequences (output is on CPU)
        output = self.embedding(sequences)

        # Apply each self-attention layer sequentially (output remains on the same device as layers)
        for attention_layer in self.self_attention_layers:
            output = attention_layer(output)

        # Reshape the output for the fully connected layer
        batch_size, num_sequences, seq_len, embed_dim = output.shape
        output = output[:, 0, :, :]  # Extract the sequence-level representation

        # Apply the fully connected layer to get class scores for each position
        output = self.fc(output)  # Output remains on the same device
        return output  # Shape: [batch_size, seq_len, num_classes]

class SAPTModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers):
        super().__init__()
        print(f"Initializing SAPTModel with {num_layers} self-attention layers...")

        self.seq_len = seq_len
        self.num_layers = num_layers

        # Embedding layer for sequences
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multiple self-attention blocks, each with its own parameters
        self.self_attention_layers = nn.ModuleList([
            MSASelfAttentionBlock(dim=embed_dim, seq_len=seq_len, heads=num_heads, dim_head=64, dropout=0.3)
            for _ in range(num_layers)
        ])

        # Fully connected layers for SAPT prediction
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.fc2 = nn.Linear(embed_dim // 2, 1)  # Output one scalar value per sequence

    def forward(self, sequences):
        # Assume input tensors are on the correct device

        # Get the embeddings for the input sequences
        output = self.embedding(sequences)

        # Apply each self-attention layer sequentially
        for attention_layer in self.self_attention_layers:
            output = attention_layer(output)

        # Reshape the output to aggregate embeddings across the sequence dimension
        # Average pooling across the sequence dimension (dim=2)
        output = output.mean(dim=2)  # Shape: (batch_size, num_sequences, embed_dim)

        # Pass through the fully connected layers
        output = self.fc1(output)
        output = torch.relu(output)
        output = self.fc2(output).squeeze(-1)  # Shape: (batch_size)

        return output


class UnifiedModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers, num_classes_ss=8, num_classes_pt=2):
        super().__init__()
        print(f"Initializing UnifiedModel with {num_layers} unique self-attention layers...")

        self.seq_len = seq_len
        self.num_layers = num_layers

        # Shared Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multiple self-attention blocks, each with its own parameters
        self.self_attention_layers = nn.ModuleList([
            MSASelfAttentionBlock(dim=embed_dim, seq_len=seq_len, heads=num_heads, dim_head=64, dropout=0.3)
            for _ in range(num_layers)
        ])

        # Separate Fully Connected layers for each task
        self.fc_sasa = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)  # Output one value per sequence position for SASA
        )

        self.fc_ss = nn.Linear(embed_dim, num_classes_ss)  # Classification layer for Secondary Structure
        self.fc_pt = nn.Linear(embed_dim, num_classes_pt)  # Classification layer for Paratope

    def forward(self, sequences):
        # Shared embedding layer
        output = self.embedding(sequences)

        # Apply each self-attention layer sequentially
        for layer in self.self_attention_layers:
            output = layer(output)

        # Reshape output for fully connected layers
        batch_size, num_sequences, seq_len, embed_dim = output.shape
        output = output[:, 0, :, :]  # Assuming you want to use the first sequence position

        # Task-specific fully connected layers
        sasa_output = self.fc_sasa(output).squeeze(-1)  # SASA output: [batch_size, seq_len]
        ss_output = self.fc_ss(output)  # Secondary Structure output: [batch_size, seq_len, num_classes_ss]
        pt_output = self.fc_pt(output)  # Paratope output: [batch_size, seq_len, num_classes_pt]

        return sasa_output, ss_output, pt_output

class PTSSModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers, num_classes_ss=8, num_classes_pt=2):
        super().__init__()
        print(f"Initializing PTSSModel with {num_layers} self-attention layers...")

        self.seq_len = seq_len
        self.num_layers = num_layers

        # Shared Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multiple self-attention blocks
        self.self_attention_layers = nn.ModuleList([
            MSASelfAttentionBlock(dim=embed_dim, seq_len=seq_len, heads=num_heads, dim_head=64, dropout=0.3)
            for _ in range(num_layers)
        ])

        # Task-specific Fully Connected layers
        self.fc_ss = nn.Linear(embed_dim, num_classes_ss)  # Classification layer for Secondary Structure
        self.fc_pt = nn.Linear(embed_dim, num_classes_pt)  # Classification layer for Paratope

    def forward(self, sequences):
        # Shared embedding layer
        output = self.embedding(sequences)

        # Apply self-attention layers
        for layer in self.self_attention_layers:
            output = layer(output)

        # Reshape output for fully connected layers
        batch_size, num_sequences, seq_len, embed_dim = output.shape
        output = output[:, 0, :, :]  # Assuming you use the first sequence position

        # Task-specific outputs
        ss_output = self.fc_ss(output)  # Secondary Structure output: [batch_size, seq_len, num_classes_ss]
        pt_output = self.fc_pt(output)  # Paratope output: [batch_size, seq_len, num_classes_pt]

        return ss_output, pt_output

class PTSASAModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers, num_classes_pt=2):
        super().__init__()
        print(f"Initializing PTSASAModel with {num_layers} self-attention layers...")

        self.seq_len = seq_len
        self.num_layers = num_layers

        # Shared Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multiple self-attention blocks
        self.self_attention_layers = nn.ModuleList([
            MSASelfAttentionBlock(dim=embed_dim, seq_len=seq_len, heads=num_heads, dim_head=64, dropout=0.3)
            for _ in range(num_layers)
        ])

        # Task-specific Fully Connected layers
        self.fc_sasa = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)  # SASA output: [batch_size, seq_len]
        )
        self.fc_pt = nn.Linear(embed_dim, num_classes_pt)  # Classification layer for Paratope

    def forward(self, sequences):
        # Shared embedding layer
        output = self.embedding(sequences)

        # Apply self-attention layers
        for layer in self.self_attention_layers:
            output = layer(output)

        # Reshape output for fully connected layers
        batch_size, num_sequences, seq_len, embed_dim = output.shape
        output = output[:, 0, :, :]  # Assuming you use the first sequence position

        # Task-specific outputs
        sasa_output = self.fc_sasa(output).squeeze(-1)  # SASA output: [batch_size, seq_len]
        pt_output = self.fc_pt(output)  # Paratope output: [batch_size, seq_len, num_classes_pt]

        return sasa_output, pt_output

class PTSAPTModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers, num_classes_pt=2):
        super().__init__()
        print(f"Initializing PTSAPTModel with {num_layers} self-attention layers...")

        self.seq_len = seq_len
        self.num_layers = num_layers

        # Shared Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multiple self-attention blocks
        self.self_attention_layers = nn.ModuleList([
            MSASelfAttentionBlock(dim=embed_dim, seq_len=seq_len, heads=num_heads, dim_head=64, dropout=0.3)
            for _ in range(num_layers)
        ])

        # Task-specific Fully Connected layers
        self.fc_sapt = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)  # SAPT output: [batch_size]
        )
        self.fc_pt = nn.Linear(embed_dim, num_classes_pt)  # Classification layer for Paratope

    def forward(self, sequences):
        # Shared embedding layer
        output = self.embedding(sequences)

        # Apply self-attention layers
        for layer in self.self_attention_layers:
            output = layer(output)

        # Reshape output for fully connected layers
        batch_size, num_sequences, seq_len, embed_dim = output.shape
        output = output[:, 0, :, :]  # Assuming you use the first sequence position

        # Task-specific outputs
        sapt_output = self.fc_sapt(output).squeeze(-1)  # SAPT output: [batch_size]
        pt_output = self.fc_pt(output)  # Paratope output: [batch_size, seq_len, num_classes_pt]

        return sapt_output, pt_output




class InteractiveModel(nn.Module):
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
        # Assume that input tensors are already on the correct device
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

        # Basic model for embedding and attention
        self.basic_model = Interactive(vocab_size, seq_len, embed_dim, num_heads, num_layers)

        # Fully connected layer to output per-residue prediction (or per-sequence if using pooling)
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sequences1, sequences2):
        # Pass the input sequences through the basic model
        output1, output2 = self.basic_model(sequences1, sequences2)

        # Reshape as needed for per-residue or pooled predictions
        output1 = output1[:, 0, :, :]  # Confirm this is intended
        output2 = output2[:, 0, :, :]

        # Apply fully connected layers to generate per-residue predictions
        prediction1 = self.fc(output1)  # Shape (batch_size, seq_len, 1)
        prediction2 = self.fc(output2)  # Shape (batch_size, seq_len, 1)

        return prediction1, prediction2

    def predict(self, sequences1, sequences2):
        # Forward pass to get logits
        logits1, logits2 = self.forward(sequences1, sequences2)

        # Sigmoid for probabilities
        probs1 = self.sigmoid(logits1)
        probs2 = self.sigmoid(logits2)

        # Threshold for binary classification (if applicable)
        bool_output1 = probs1 >= 0.5
        bool_output2 = probs2 >= 0.5

        return bool_output1, bool_output2

class MSAModelWithGNN(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers, num_gnn_layers):
        super().__init__()
        print(f"Initializing MSAModel with {num_gnn_layers} GNN layers and Residual Connections...")

        # Basic model for embedding and attention
        self.interactive_model = Interactive(vocab_size, seq_len, embed_dim, num_heads, num_layers)

        # Separate fully connected layers for each sequence
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)

        # GNN layers for each sequence
        self.gnn_layers1 = nn.ModuleList([GCNConv(embed_dim, embed_dim) for _ in range(num_gnn_layers)])
        self.gnn_layers2 = nn.ModuleList([GCNConv(embed_dim, embed_dim) for _ in range(num_gnn_layers)])

        # Layer normalization for residual connections after each GNN layer
        self.norm_layers1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_gnn_layers)])
        self.norm_layers2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_gnn_layers)])

        # Final output layers (optional)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sequences1, sequences2, adjacency_matrix1, adjacency_matrix2):
        # Pass the input sequences through the basic model
        output1, output2 = self.interactive_model(sequences1, sequences2)

        # Reshape outputs for per-residue predictions
        output1 = output1[:, 0, :, :]  # Shape: (batch_size, seq_len, embed_dim)
        output2 = output2[:, 0, :, :]  # Shape: (batch_size, seq_len, embed_dim)

        # Apply fully connected layers with residual connection
        embedding1 = self.fc1(output1) + output1  # Residual connection
        embedding2 = self.fc2(output2) + output2  # Residual connection

        # Flatten embeddings for GNN (required format: (num_nodes, embed_dim))
        batch_size, seq_len, embed_dim = embedding1.size()
        embedding1_flat = embedding1.view(-1, embed_dim)  # Shape: (batch_size * seq_len, embed_dim)
        embedding2_flat = embedding2.view(-1, embed_dim)

        # Convert adjacency matrices to edge index format if necessary
        edge_index1 = adjacency_matrix1.nonzero(as_tuple=False).t()  # Shape: (2, num_edges)
        edge_index2 = adjacency_matrix2.nonzero(as_tuple=False).t()  # Shape: (2, num_edges)

        # Pass through multiple GNN layers with residual connections and normalization
        for gnn_layer1, gnn_layer2, norm1, norm2 in zip(self.gnn_layers1, self.gnn_layers2, self.norm_layers1, self.norm_layers2):
            gnn_out1 = gnn_layer1(embedding1_flat, edge_index1)  # Shape: (num_nodes, embed_dim)
            gnn_out2 = gnn_layer2(embedding2_flat, edge_index2)

            # Residual connections and normalization
            embedding1_flat = norm1(gnn_out1 + embedding1_flat)
            embedding2_flat = norm2(gnn_out2 + embedding2_flat)

        # Reshape back to batch format
        gnn_output1 = embedding1_flat.view(batch_size, seq_len, embed_dim)
        gnn_output2 = embedding2_flat.view(batch_size, seq_len, embed_dim)

        # Final predictions (optional sigmoid or additional layers)
        prediction1 = self.sigmoid(gnn_output1)  # Shape: (batch_size, seq_len, embed_dim)
        prediction2 = self.sigmoid(gnn_output2)  # Shape: (batch_size, seq_len, embed_dim)

        return prediction1, prediction2


class MRPModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers):
        print("Initializing MRPModel...")
        super().__init__()
        print("Finished initializing MRPModel...")
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.MSASelfAttention = MSASelfAttentionBlock(dim=embed_dim, seq_len=seq_len, heads=num_heads, dim_head=64, dropout=0.3)
        self.fc = nn.Linear(embed_dim, vocab_size)  # Output a probability distribution over the vocab size

    def forward(self, sequences1, mask_pos1=None):
        batch_size, num_seqs, seq_len = sequences1.shape[:3]
 
        # Mask the original sequences (first sequence along dim=1) by setting the embedding at the mask position to the padding index (0)
        if mask_pos1 is not None:
            sequences1 = sequences1.clone()
            print("seq.shape",sequences1.shape)
            sequences1[torch.arange(batch_size), 0, mask_pos1] = 0  # Masking with the padding token (index 0)
 
        # Embedding for the input sequences
        embedded1 = self.embedding(sequences1)  
        print("embed_shape", embedded1.shape)
        # Apply self-attention to each sequence (only affecting the sequence dimension)
        output1 = self.MSASelfAttention(embedded1)  
        print("output1_shape",output1.shape)
        # Gather the outputs at the masked positions from the original sequence (index 0 along num_seqs)
        if mask_pos1 is not None:
            # Ensure mask positions are within the valid sequence length
            output1 = output1[torch.arange(batch_size), 0, mask_pos1, :]
            # Apply the fully connected layer to predict the masked residue
            prediction1 = self.fc(output1)  
 
            return prediction1
        else:
            raise ValueError("mask_pos1 and mask_pos2 must be provided for masking.")

    def predict(self, sequences1, mask_pos1=None):
        # Get the logits from the forward pass, using masked positions if provided
        logits1, logits2 = self.forward(sequences1, mask_pos1)

        # Apply softmax to get probabilities
        probs1 = nn.functional.softmax(logits1, dim=-1)  # [batch_size, vocab_size]

        # Get the predicted class (residue) indices
        pred_residues1 = torch.argmax(probs1, dim=-1)  # [batch_size]

        return pred_residues1
