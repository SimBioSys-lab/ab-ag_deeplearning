import torch, esm
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import einsum, nn
from torch_geometric.nn import GATConv
from einops import rearrange, repeat
import math
from Dynamic_Mask import DyM
from timm.models.layers import DropPath  # or your own DropPath impl
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

    def forward(self, x, tied=False, return_attention=False):
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
            scaling_factor = math.sqrt(dots.size(0))  # Consider verifying if this is the intended scaling
            dots = (rowwise_average / scaling_factor).expand_as(dots)
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

    def forward(self, x, return_attention=False, tied=False):
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
            tied = tied
        x = rearrange(x, input_fold_eq)

        # Forward pass through attention, optionally retrieving attention scores
        if return_attention:
            out, attn = self.attn(x, tied=tied, return_attention=True)
        else:
            out = self.attn(x, tied=tied, return_attention=False)
            attn = None

        out = rearrange(out, output_fold_eq, h=h, w=w)
        return (out, attn) if return_attention else out


class MSASelfAttentionBlock(nn.Module):
    def __init__(self, dim, seq_len, heads, dim_head, dropout=0.0):
        super().__init__()
        self.row_attn = AxialAttention(dim=dim, heads=heads, dropout=dropout, row_attn=True, col_attn=False)
        self.col_attn = AxialAttention(dim=dim, heads=heads, dropout=dropout, row_attn=False, col_attn=True)
    def forward(self, x):
        x = self.row_attn(x)
        x = self.col_attn(x)
        return x

_AA1_TO_ESM = {
    'A': 21, 'R':  4, 'N':  3, 'D':  6, 'C': 13,
    'Q':  8, 'E':  9, 'G': 16, 'H':  1, 'I': 14,
    'L':  7, 'K': 15, 'M': 11, 'F': 10, 'P':  5,
    'S': 19, 'T':  2, 'W': 12, 'Y': 20, 'V': 18,
    'X': 22,           # unknown
    '-':  0,           # gap
    '0': 23            # PAD (we use ASCII ‘0’ to mark padding)
}
_LUT = torch.tensor([_AA1_TO_ESM.get(chr(i), 22) for i in range(256)],
                    dtype=torch.long)            # (256,)

# ──────────────────────────────────────────────────────────────
class CoreModel(nn.Module):
    """
    Frozen (or fine-tuned) ESM-2 embeddings  ➟  Axial MSA  ➟  DyM + FFN
    Input : uint8 ASCII tensor (B, 64, L)
    Output: float32 tensor     (B, 64, L, 320)
    """

    def __init__(
        self,
        vocab_size      : int,           # kept for compatibility (unused)
        seq_len         : int,
        embed_dim       : int = None,    # ignored – always 320
        num_heads       : int = 16,
        dropout         : float = 0.1,
        num_layers      : int = 1,
        drop_path_rate  : float = 0.1,
        esm_ckpt        : str  = "esm2_t6_8M_UR50D",
        freeze_esm      : bool = True,
    ):
        super().__init__()
        # 1)  ESM-2
        self.esm, _ = esm.pretrained.load_model_and_alphabet_hub(esm_ckpt)
        self.embed_dim = self.esm.embed_dim     # 320
        self.seq_len   = seq_len

        if freeze_esm:
            self.esm.eval()
            for p in self.esm.parameters():
                p.requires_grad_(False)

        # 2)  Down-stream blocks (dim = 320)
        self.sa_blocks = nn.ModuleList([
            MSASelfAttentionBlock(dim=self.embed_dim,
                                  heads=num_heads,
                                  dim_head=64,
                                  dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ffn_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, 4 * self.embed_dim),
                nn.GELU(),
                nn.Linear(4 * self.embed_dim, self.embed_dim)
            ) for _ in range(num_layers)
        ])
        self.norm_sa  = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_layers)])
        self.norm_ffn = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)
        self.dym_sa  = nn.ModuleList([DyM(self.embed_dim) for _ in range(num_layers)])
        self.dym_ffn = nn.ModuleList([DyM(self.embed_dim) for _ in range(num_layers)])

        dpr = torch.linspace(0, drop_path_rate, num_layers).tolist()
        self.drop_sa  = nn.ModuleList([DropPath(d) if d > 0 else nn.Identity() for d in dpr])
        self.drop_ffn = nn.ModuleList([DropPath(d) if d > 0 else nn.Identity() for d in dpr])

    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _chars_to_esm(x: torch.ByteTensor) -> torch.LongTensor:
        """
        uint8 ASCII tensor → int64 ESM IDs  (vectorised LUT)
        """
        return _LUT.to(x.device)[x]

    # ──────────────────────────────────────────────────────────
    def forward(self, seq_ascii: torch.ByteTensor) -> torch.Tensor:
        """
        seq_ascii : (B, 64, L) uint8.  PAD = ASCII '0'
        returns   : (B, 64, L, 320)
        """
        B, S, L = seq_ascii.shape
        assert S == 64 and L == self.seq_len, "Unexpected input shape"

        # 1)  ASCII → token-IDs → ESM embeddings
        toks = self._chars_to_esm(seq_ascii).view(B * S, L)          # (B·64, L)
        with torch.set_grad_enabled(not self.esm.eval()):
            rep = self.esm(toks, repr_layers=[0])["representations"][0]  # (B·64,L,320)
        x = rep.view(B, S, L, self.embed_dim)                        # (B,64,L,320)

        # 2)  Layer stack
        for i in range(len(self.sa_blocks)):
            # Self-attention (Axial MSA)
            res = x
            attn = self.sa_blocks[i](x)              # (B,64,L,320)
            attn = self.dropout(attn)
            attn = self.dym_sa[i](attn)
            attn = self.drop_sa[i](attn)
            x    = self.norm_sa[i](res + attn)

            # FFN branch
            res2 = x
            ffn  = self.ffn_blocks[i](x)             # (B,64,L,320)
            ffn  = self.dropout(ffn)
            ffn  = self.dym_ffn[i](ffn)
            ffn  = self.drop_ffn[i](ffn)
            x    = self.norm_ffn[i](res2 + ffn)

        return x   # (B,64,L,320)
    
class CGModel(nn.Module):
    """
    CoreModel + GNN layers + external DyM gating + DropPath.
    """
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int,
        num_gnn_layers: int,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        # base transformer encoder
        self.core_model = CoreModel(
            vocab_size, seq_len, embed_dim, num_heads, dropout, num_layers, drop_path_rate
        )
        self.dropout = nn.Dropout(dropout)

        # positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(seq_len, embed_dim))
        nn.init.xavier_uniform_(self.pos_emb)

        # initial FC + DyM + DropPath
        self.fc      = nn.Linear(embed_dim, embed_dim)
        self.fc_dym  = DyM(embed_dim)
        self.fc_dp   = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        # GNN + FFN stacks
        self.gnn_layers = nn.ModuleList([
            GATConv(embed_dim, embed_dim // num_heads, heads=num_heads, concat=True)
            for _ in range(num_gnn_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 4*embed_dim),
                nn.GELU(),
                nn.Linear(4*embed_dim, embed_dim)
            )
            for _ in range(num_gnn_layers)
        ])

        # norms
        self.gnn_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_gnn_layers)])
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_gnn_layers)])

        # external DyM and DropPath for GNN/FFN
        self.gnn_dym = nn.ModuleList([DyM(embed_dim) for _ in range(num_gnn_layers)])
        self.ffn_dym = nn.ModuleList([DyM(embed_dim) for _ in range(num_gnn_layers)])
        self.gnn_dp  = nn.ModuleList([
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
            for _ in range(num_gnn_layers)
        ])
        self.ffn_dp  = nn.ModuleList([
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
            for _ in range(num_gnn_layers)
        ])

    def forward(self, sequences, padded_edges):
        B, _, max_e = padded_edges.shape
        seq_len = sequences.shape[2]

        # rebuild edge_index & batch idx
        edge_indices, batch_idx = [], []
        for i in range(B):
            valid = padded_edges[i][:, padded_edges[i][0] != -1]
            valid = valid + i * seq_len
            edge_indices.append(valid)
            batch_idx.append(torch.full((valid.shape[1],), i,
                                        dtype=torch.long, device=valid.device))
        edge_index = torch.cat(edge_indices, dim=1)

        # 1) core transformer
        x = self.core_model(sequences)    # (B, seq_len, D)
        x = self.dropout(x)
        x = x[:, 0, :, :]                 # (B, seq_len, D)

        # 2) add positional
        pos = self.pos_emb[:seq_len].unsqueeze(0).expand(B, -1, -1)
        x = x + self.dropout(pos)

        # 3) initial FC block
        res   = x
        fc_out = self.fc(x)
        fc_out = self.dropout(fc_out)
        fc_out = self.fc_dym(fc_out)
        fc_out = self.fc_dp(fc_out)
        x = res + fc_out

        # flatten for GNN
        x_flat = x.view(-1, x.size(-1))

        # 4) GNN + FFN blocks
        for i in range(len(self.gnn_layers)):
            # GNN branch
            res_g = x_flat
            g_out = self.gnn_layers[i](x_flat, edge_index)
            g_out = self.dropout(g_out)
            g_out = self.gnn_dym[i](g_out)
            g_out = self.gnn_dp[i](g_out)
            x_flat = self.gnn_norm[i](res_g + g_out)

            # FFN branch
            res_f = x_flat
            f_out = self.ffn_layers[i](x_flat)
            f_out = self.dropout(f_out)
            f_out = self.ffn_dym[i](f_out)
            f_out = self.ffn_dp[i](f_out)
            x_flat = self.ffn_norm[i](res_f + f_out)

        # reshape back
        refined = x_flat.view(B, 1, seq_len, -1)
        return refined

class MCModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int,
        num_gnn_layers: int,
        num_int_layers: int,
        drop_path_rate: float = 0.0,    # new argument
    ):
        super().__init__()
        print(f"Initializing MCModel with {num_gnn_layers} GAT layers and "
              f"{num_int_layers} row attention layers (drop_path_rate={drop_path_rate})")

        # your existing CGModel
        self.cg_model = CGModel(
            vocab_size, seq_len, embed_dim, num_heads,
            dropout, num_layers, num_gnn_layers, drop_path_rate
        )

        # row‐attention / FFN stacks
        self.row_attn_layers = nn.ModuleList([
            AxialAttention(dim=embed_dim, heads=num_heads,
                           dropout=dropout, row_attn=True, col_attn=False)
            for _ in range(num_int_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 4*embed_dim),
                nn.GELU(),
                nn.Linear(4*embed_dim, embed_dim)
            )
            for _ in range(num_int_layers)
        ])
        self.row_norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_int_layers)])
        self.ffn_norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_int_layers)])
        self.dropout = nn.Dropout(dropout)

        # replace masks with external DyM
        self.row_dym = nn.ModuleList([DyM(embed_dim) for _ in range(num_int_layers)])
        self.ffn_dym = nn.ModuleList([DyM(embed_dim) for _ in range(num_int_layers)])

        # stochastic depth schedule
        dpr = torch.linspace(0, drop_path_rate, num_int_layers).tolist()
        self.row_dp = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity()
            for i in range(num_int_layers)
        ])
        self.ffn_dp = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity()
            for i in range(num_int_layers)
        ])

    def forward(self, sequences, padded_edges, pairwise_repr=None,
                return_attention=False, tied=False):
        # 1) initial embeddings
        output = self.cg_model(sequences, padded_edges)
        last_attn = None

        # 2) row‐attention + FFN with DyM + DropPath
        for idx, (row_attn, ffn, rnorm, fnorm) in enumerate(zip(
                self.row_attn_layers, self.ffn_layers,
                self.row_norm_layers, self.ffn_norm_layers)):

            # row attention
            if return_attention and idx == len(self.row_attn_layers) - 1:
                row_out, attn = row_attn(output, return_attention=True, tied=tied)
                last_attn = attn
            else:
                row_out = row_attn(output)

            row_out = self.dropout(row_out)
            row_out = self.row_dym[idx](row_out)    # DyM on every token
            row_out = self.row_dp[idx](row_out)     # DropPath
            output  = rnorm(output + row_out)       # residual + norm

            # feed‐forward
            ffn_out = ffn(output)
            ffn_out = self.dropout(ffn_out)
            ffn_out = self.ffn_dym[idx](ffn_out)    # DyM
            ffn_out = self.ffn_dp[idx](ffn_out)     # DropPath
            output  = fnorm(output + ffn_out)       # residual + norm

        if return_attention:
            return output, last_attn
        return output

class ClassificationModel(nn.Module):
    """
    Classification model using MCModel and a classification layer.
    """
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, dropout, 
                 num_layers, num_gnn_layers, num_classes, num_int_layers, drop_path_rate):
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
                                num_layers, num_gnn_layers, num_int_layers, drop_path_rate)

        # Final classification layer maps the refined embedding to the desired number of classes.
        self.fc_classification = nn.Linear(embed_dim, num_classes)

    def forward(self, sequences, padded_edges, return_attention=False, tied=False):
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
            refined_embedding, last_attn = self.mc_model(sequences, padded_edges, return_attention=return_attention, tied=tied)
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
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, dropout, num_layers, num_gnn_layers, num_int_layers, drop_path_rate):
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
        self.mc_model = MCModel(vocab_size, seq_len, embed_dim, num_heads, dropout, num_layers, num_gnn_layers, num_int_layers, drop_path_rate)

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

class CTMModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, dropout,
                 num_layers, num_gnn_layers, num_classes, num_int_layers, drop_path_rate):
        """
        Args:
            vocab_size (int): Vocabulary size for input sequences.
            seq_len (int): Length of the input sequences.
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            num_layers (int): Number of transformer (or axial) layers.
            num_gnn_layers (int): Number of graph neural network layers.
            num_classes (int): Number of output classes. For binary contact prediction, use 2.
            num_int_layers (int): Number of intermediate layers (if any) in MCModel.
        """
        super().__init__()
        print(f"Initializing CTMModel with {num_gnn_layers} GAT layers and residual connections...")

        # Instantiate the MCModel (assumed to return refined embeddings and attention scores)
        self.mc_model = MCModel(vocab_size, seq_len, embed_dim, num_heads, dropout,
                                num_layers, num_gnn_layers, num_int_layers, drop_path_rate)

        # Since we are not averaging over heads, the attention tensor remains
        # with shape [B, num_heads, seq_len, seq_len]. The Conv2d layer maps from
        # num_heads input channels to num_classes output channels.
        self.fc_linear = nn.Linear(num_heads, num_classes)

    def forward(self, sequences, padded_edges, return_attention=True, tied=False):
        """
        Args:
            sequences (torch.Tensor): Input sequences of shape [batch_size, seq_len, feature_dim].
            padded_edges (torch.Tensor): Padded edge indices of shape [batch_size, 2, max_edges].
            return_attention (bool): Whether to also return the intermediate attention scores.
            tied (bool): An optional flag passed to MCModel (if needed).
        Returns:
            If return_attention is False:
                torch.Tensor: Logits of shape [batch_size, num_classes, seq_len, seq_len].
            Else:
                Tuple[torch.Tensor, torch.Tensor]: (logits, last_attn)
                    - logits: as above.
                    - last_attn: raw attention scores from MCModel with shape [batch_size, num_heads, seq_len, seq_len].
        """
        # Get refined embeddings and attention scores from MCModel.
        # We assume MCModel returns a tuple: (refined_embedding, last_attn)
        # where last_attn has shape [batch_size, num_heads, seq_len, seq_len]
        refined_embedding, last_attn = self.mc_model(sequences, padded_edges, return_attention=return_attention, tied=tied)
        x = last_attn.permute(0, 2, 3, 1)
        x = self.fc_linear(x)
        logits = x.permute(0, 3, 1, 2)
        if return_attention:
            return logits, last_attn
        else:
            return logits
