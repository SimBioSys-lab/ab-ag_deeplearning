import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import einsum, nn
from torch_geometric.nn import GATConv
from einops import rearrange, repeat
import math

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
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.gating = nn.Linear(dim, inner_dim)

        # Initialize gating layer so that initially gates are ~1.
        nn.init.constant_(self.gating.weight, 0.0)
        nn.init.constant_(self.gating.bias, 1.0)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None, tied=False, return_attention=False):
        h = self.heads

        # Generate query, key, and value
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # Scale the query
        q = q * self.scale

        # Compute dot-product attention scores
        dots = einsum("b h i d, b h j d -> b h i j", q, k)

        # Optionally apply tied attention (averaging along the row dimension)
        if tied:
            rowwise_average = torch.mean(dots, dim=3, keepdim=True)
            b_sqrt = math.sqrt(dots.size(0))
            dots = rowwise_average / b_sqrt
            dots = rowwise_average.expand_as(dots)

        # Softmax to get probabilities
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # Compute weighted sum of values
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        # Apply gating mechanism
        gates = self.gating(x)
        out = out * gates.sigmoid()

        # Final projection and residual connection
        out = self.to_out(out)
        out = x + out
        out = self.norm(out)

        # Return both output and attention weights if requested
        if return_attention:
            return out, attn
        return out

class AxialAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0, row_attn=True, col_attn=True, **kwargs):
        super().__init__()
        assert row_attn or col_attn, "Either row or column attention must be turned on."
        self.row_attn = row_attn
        self.col_attn = col_attn
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, heads=heads, dropout=dropout, **kwargs)

    def forward(self, x, edges=None, mask=None, return_attention=False):
        # Ensure that only one type of axial attention is used
        assert self.row_attn ^ self.col_attn, "Has to be either row or column attention, not both."

        b, h, w, d = x.shape
        x = self.norm(x)

        # Choose the proper folding based on whether we're attending along rows or columns
        if self.col_attn:
            # Process columns: treat width (w) as the sequence length
            input_fold_eq = "b h w d -> (b w) h d"
            output_fold_eq = "(b w) h d -> b h w d"
            tied = False
        elif self.row_attn:
            # Process rows: treat height (h) as the sequence length
            input_fold_eq = "b h w d -> (b h) w d"
            output_fold_eq = "(b h) w d -> b h w d"
            tied = True

        x = rearrange(x, input_fold_eq)

        # Forward pass through attention, optionally retrieving attention scores
        if return_attention:
            out, attn = self.attn(x, mask=mask, tied=tied, return_attention=True)
        else:
            out = self.attn(x, mask=mask, tied=tied, return_attention=False)
            attn = None

        out = rearrange(out, output_fold_eq, h=h, w=w)
        return (out, attn) if return_attention else out


class MSASelfAttentionBlock(nn.Module):
    def __init__(self, dim, seq_len, heads, dim_head, dropout=0.0):
        super().__init__()
        self.row_attn = AxialAttention(dim=dim, heads=heads, dropout=dropout, row_attn=True, col_attn=False)
        self.col_attn = AxialAttention(dim=dim, heads=heads, dropout=dropout, row_attn=False, col_attn=True)

    def forward(self, x, mask=None, pairwise_repr=None):
        x = self.row_attn(x, mask=mask, edges=pairwise_repr)
        x = self.col_attn(x, mask=mask)
        return x

class CoreModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, dropout, num_layers):
        super().__init__()
        print(f"Initializing CoreModel with {num_layers} self-attention layers...")

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.self_attention_layers = nn.ModuleList([
            MSASelfAttentionBlock(dim=embed_dim, seq_len=seq_len, heads=num_heads, dim_head=64, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])

    def forward(self, sequences):
        # Embedding layer
        output = self.embedding(sequences)

        # Self-attention layers with residual connections
        for attention_layer, norm_layer in zip(self.self_attention_layers, self.norm_layers):
            residual = output  # Save the input as residual
            output = attention_layer(output)
            output = norm_layer(output + residual)  # Add residual and normalize

        return output

class CGModel(nn.Module):
    """
    CoreModel with GNN layers encapsulated. Outputs refined embeddings.
    """
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, dropout, num_layers, num_gnn_layers):
        """
        Args:
            vocab_size (int): Vocabulary size for the CoreModel.
            seq_len (int): Length of input sequences.
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads in the GAT layers.
            num_layers (int): Number of layers in the CoreModel.
            num_gnn_layers (int): Number of GNN layers.
        """
        super().__init__()
        print(f"Initializing CGModel with {num_gnn_layers} GAT layers and residual connections...")

        # Core Model for sequence embeddings
        self.core_model = CoreModel(vocab_size, seq_len, embed_dim, num_heads, dropout, num_layers)

        self.dropout = nn.Dropout(dropout)

        # Positional embedding
        self.positional_embeddings = nn.Parameter(torch.zeros(seq_len, embed_dim))
        nn.init.xavier_uniform_(self.positional_embeddings)  # Initialize weights

        # Fully connected layer for initial embedding transformation
        self.fc = nn.Linear(embed_dim, embed_dim)

        # GAT layers for graph-based sequence refinement
        self.gnn_layers = nn.ModuleList(
            [GATConv(embed_dim, embed_dim, heads=1, concat=False) for _ in range(num_gnn_layers)]
        )
        # Layer normalization for residual connections after each GAT layer
        self.norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_gnn_layers)])

    def forward(self, sequences, padded_edges):
        """
        Forward pass for CGModel with offset correction for batched edges.
        Args:
            padded_edges (torch.Tensor): Padded edge indices of shape [batch_size, 2, max_edges].
            sequences (torch.Tensor): Input sequences of shape [batch_size, seq_len, feature_dim].
        Returns:
            torch.Tensor: Refined embeddings of shape [batch_size, seq_len, embed_dim].
        """
        batch_size, _, max_edges = padded_edges.shape
        seq_len = sequences.shape[2]  # Assuming all graphs have the same number of nodes

        # Recover original edge indices by filtering out `-1`s
        edge_indices = []
        batch = []
        for i in range(batch_size):
            valid_edges = padded_edges[i][:, padded_edges[i][0] != -1]  # Filter out `-1`s
            valid_edges = valid_edges + i * seq_len  # Adjust for batch offset
            edge_indices.append(valid_edges)
            batch.append(torch.full((valid_edges.shape[1],), i, dtype=torch.long, device=valid_edges.device))

        # Concatenate edges and create batched graph
        edge_index = torch.cat(edge_indices, dim=1)  # Concatenate edges
        batch = torch.cat(batch, dim=0)  # Concatenate batch indices

        # Pass the input sequences through the CoreModel
        output = self.core_model(sequences)  # Shape: [batch_size, num_seq, seq_len, embed_dim]
        output = self.dropout(output)
        output = output[:, 0, :, :]  # Extract the first sequence: [batch_size, seq_len, embed_dim]

        # Add positional embeddings
        positional_embeddings = self.positional_embeddings[:seq_len, :].unsqueeze(0)  # Shape: (1, seq_len, embed_dim)
        positional_embeddings = positional_embeddings.expand(batch_size, -1, -1)  # Expand to batch size
        output = output + self.dropout(positional_embeddings)


        # Apply the fully connected layer with a residual connection
        embedding = self.fc(output) + output  # Residual connection
        embedding_flat = embedding.view(-1, embedding.shape[-1])  # Flatten embeddings for GNN input

        # Pass through GNN layers
        for gnn_layer, norm in zip(self.gnn_layers, self.norm_layers):
            gnn_out = gnn_layer(embedding_flat, edge_index)
            gnn_out = self.dropout(gnn_out)
            embedding_flat = norm(gnn_out + embedding_flat)

        # Reshape back to batch format
        refined_embedding = embedding_flat.view(batch_size, 1, seq_len, embedding_flat.shape[-1])  # Shape: [batch_size, 1, seq_len, embed_dim]
        return refined_embedding

class MCModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, dropout, 
                 num_layers, num_gnn_layers, num_int_layers):
        super().__init__()
        print(f"Initializing MCModel with {num_gnn_layers} GAT layers and {num_int_layers} row attention layers...")

        self.cg_model = CGModel(vocab_size, seq_len, embed_dim, num_heads, dropout, num_layers, num_gnn_layers)

        self.row_attn_layers = nn.ModuleList([
            AxialAttention(dim=embed_dim, heads=num_heads, dropout=dropout, row_attn=True, col_attn=False)
            for _ in range(num_int_layers)
        ])

        self.row_norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_int_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequences, padded_edges, mask=None, pairwise_repr=None, return_attention=False):
        # Core model forward pass
        output = self.cg_model(sequences, padded_edges)
        last_attn = None  # will hold attention from the last axial attention layer

        # Row attention refinement
        for idx, (row_attn, norm) in enumerate(zip(self.row_attn_layers, self.row_norm_layers)):
            # If we are in the last axial attention layer and want to export attention scores:
            if return_attention and idx == len(self.row_attn_layers) - 1:
                # Here we assume that AxialAttention.forward accepts `return_attention`
                row_out, attn = row_attn(output, mask=mask, edges=None, return_attention=True)
                last_attn = attn
            else:
                row_out = row_attn(output, mask=mask, edges=None)
            row_out = self.dropout(row_out)
            output = norm(row_out + output)

        if return_attention:
            return output, last_attn
        return output

class ClassificationModel(nn.Module):
    """
    Classification model using MCModel and a classification layer.
    """
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, dropout, 
                 num_layers, num_gnn_layers, num_classes, num_int_layers):
        """
        Args:
            vocab_size (int): Vocabulary size for the CoreModel.
            seq_len (int): Length of input sequences.
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads in the GAT layers.
            dropout (float): Dropout rate.
            num_layers (int): Number of layers in the CoreModel.
            num_gnn_layers (int): Number of GNN layers.
            num_classes (int): Number of output classes for final classification.
            num_int_layers (int): Number of internal (axial attention) layers.
        """
        super().__init__()
        print(f"Initializing ClassificationModel with {num_gnn_layers} GAT layers and residual connections...")

        # Instantiate the MCModel (which is assumed to have been modified to optionally export attention scores)
        self.mc_model = MCModel(vocab_size, seq_len, embed_dim, num_heads, dropout, 
                                num_layers, num_gnn_layers, num_int_layers)

        # Final classification layer maps the refined embedding to the desired number of classes.
        self.fc_classification = nn.Linear(embed_dim, num_classes)

    def forward(self, sequences, padded_edges, return_attention=False):
        """
        Args:
            sequences (torch.Tensor): Input sequences of shape [batch_size, seq_len, feature_dim].
            padded_edges (torch.Tensor): Padded edge indices of shape [batch_size, 2, max_edges].
            return_attention (bool): If True, export the attention scores from the last axial attention layer.
        Returns:
            If return_attention is False:
                torch.Tensor: Predictions of shape [batch_size, seq_len, num_classes].
            Else:
                Tuple[torch.Tensor, torch.Tensor]: (predictions, last_attention_scores)
        """
        # Get refined embeddings from MCModel.
        # If return_attention is True, MCModel is expected to return a tuple (refined_embedding, last_attn)
        if return_attention:
            refined_embedding, last_attn = self.mc_model(sequences, padded_edges, return_attention=True)
        else:
            refined_embedding = self.mc_model(sequences, padded_edges)

        # Final classification layer: apply a linear layer to the refined embeddings.
        predictions = self.fc_classification(refined_embedding)

        if return_attention:
            return predictions, last_attn
        return predictions

class RegressionModel(nn.Module):
    """
    Regression model using MCModel and a regression layer.
    """
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, dropout, num_layers, num_gnn_layers, num_int_layers):
        """
        Args:
            vocab_size (int): Vocabulary size for the CoreModel.
            seq_len (int): Length of input sequences.
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads in the GAT layers.
            num_layers (int): Number of layers in the CoreModel.
            num_gnn_layers (int): Number of GNN layers.
        """
        super().__init__()
        print(f"Initializing RegressionModel with {num_gnn_layers} GAT layers and residual connections...")

        # MCModel for embeddings
        self.mc_model = MCModel(vocab_size, seq_len, embed_dim, num_heads, dropout, num_layers, num_gnn_layers, num_int_layers)

        # Final regression layer
        self.fc_regression = nn.Linear(embed_dim, 1)  # Single output for regression

    def forward(self, sequences, padded_edges):
        """
        Args:
            sequences (torch.Tensor): Input sequences of shape [batch_size, seq_len, feature_dim].
            padded_edges (torch.Tensor): Padded edge indices of shape [batch_size, 2, max_edges].

        Returns:
            torch.Tensor: Regression outputs of shape [batch_size, seq_len, 1].
        """
        # Get refined embeddings from MCModel
        refined_embedding = self.mc_model(sequences, padded_edges)  # Shape: [batch_size, seq_len, embed_dim]

        # Final regression output
        predictions = torch.relu(self.fc_regression(refined_embedding))  # Shape: [batch_size, seq_len, 1]
        return predictions
