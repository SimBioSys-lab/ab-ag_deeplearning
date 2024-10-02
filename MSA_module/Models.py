import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import einsum, nn
from einops import rearrange, repeat


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

    def forward(self, x, mask=None, attn_bias=None):
        h = self.heads

        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # Scale
        q = q * self.scale

        # Attention
        dots = einsum("b h i d, b h j d -> b h i j", q, k)
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
    def __init__(self, dim, heads, row_attn=True, col_attn=True, accept_edges=False, **kwargs):
        super().__init__()
        assert row_attn or col_attn, "Either row or column attention must be turned on."

        self.row_attn = row_attn
        self.col_attn = col_attn
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, heads=heads, **kwargs)

        self.edges_to_attn_bias = (
            nn.Sequential(nn.Linear(dim, heads, bias=False), Rearrange("b i j h -> b h i j")) if accept_edges else None
        )

    def forward(self, x, edges=None, mask=None):
        assert self.row_attn ^ self.col_attn, "Has to be either row or column attention, not both."

        b, h, w, d = x.shape
        x = self.norm(x)

        # Axial attention setup
        if self.col_attn:
            axial_dim = w
            input_fold_eq = "b h w d -> (b w) h d"
            output_fold_eq = "(b w) h d -> b h w d"
        elif self.row_attn:
            axial_dim = h
            input_fold_eq = "b h w d -> (b h) w d"
            output_fold_eq = "(b h) w d -> b h w d"

        x = rearrange(x, input_fold_eq)

        if exists(mask):
            mask = rearrange(mask, input_fold_eq.replace("d", ""))

        attn_bias = None
        if exists(self.edges_to_attn_bias) and exists(edges):
            attn_bias = self.edges_to_attn_bias(edges)
            attn_bias = repeat(attn_bias, "b h i j -> (b x) h i j", x=axial_dim)

        out = self.attn(x, mask=mask, attn_bias=attn_bias)
        return rearrange(out, output_fold_eq, h=h, w=w)


class MSASelfAttentionBlock(nn.Module):
    def __init__(self, dim, seq_len, heads, dim_head, dropout=0.0):
        super().__init__()
        self.row_attn = AxialAttention(dim=dim, heads=heads, row_attn=True, col_attn=False, accept_edges=True)
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

        return mat1_to_mat2_attention, mat2_to_mat1_attention

class MSAModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers):
        print("Initializing MSAModel...")
        super().__init__()
        print("Finished initializing MSAModel...")
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.MSASelfAttention = MSASelfAttentionBlock(dim=embed_dim, seq_len=seq_len, heads=num_heads, dim_head=64, dropout=0.3)
        self.MSABidirectionalCrossAttention = MSABidirectionalCrossAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.fc = nn.Linear(embed_dim * seq_len, seq_len)  # Output same number of values as input
        self.sigmoid = nn.Sigmoid()  # Sigmoid for probability conversion

    def forward(self, sequences1, sequences2):
        batch_size = sequences1.shape[0]
        embedded1 = self.embedding(sequences1)
        embedded2 = self.embedding(sequences2)

        output1 = self.MSASelfAttention(embedded1)
        output2 = self.MSASelfAttention(embedded2)

        output1, output2 = self.MSABidirectionalCrossAttention(output1, output2)

        # Reshape outputs to be (batch_size, embed_dim * seq_len)
        output1 = output1[:, 0, :, :].reshape(batch_size, -1)
        output2 = output2[:, 0, :, :].reshape(batch_size, -1)

        # Output same number of values as input for each sequence
        prediction1 = self.fc(output1)
        prediction2 = self.fc(output2)

        return prediction1, prediction2

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
