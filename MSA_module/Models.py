import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import einsum, nn
from torch.utils.checkpoint import checkpoint_sequential


# helpers


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# attention
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        gating=True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.seq_len = seq_len
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

    def forward(
        self,
        x,
        mask=None,
        attn_bias=None,
    ):
        device, orig_shape, h = (
            x.device,
            x.shape,
            self.heads,
        )

        q, k, v = (
            self.to_q(x),
            *self.to_kv(x).chunk(2, dim=-1),
        )

        i, j = q.shape[-2], k.shape[-2]

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
            (q, k, v),
        )

        # scale

        q = q * self.scale

        # query / key similarities

        dots = einsum("b h i d, b h j d -> b h i j", q, k)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # aggregate

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")

        # gating

        gates = self.gating(x)
        out = out * gates.sigmoid()

        # combine to out

        out = self.to_out(out)
        return out

class AxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        row_attn: bool = True,
        col_attn: bool = True,
        accept_edges: bool = False,
        global_query_attn: bool = False,
        **kwargs,
    ):
        """
        Axial Attention module.

        Args:
            dim (int): The input dimension.
            heads (int): The number of attention heads.
            row_attn (bool, optional): Whether to perform row attention. Defaults to True.
            col_attn (bool, optional): Whether to perform column attention. Defaults to True.
            accept_edges (bool, optional): Whether to accept edges for attention bias. Defaults to False.
            global_query_attn (bool, optional): Whether to perform global query attention. Defaults to False.
            **kwargs: Additional keyword arguments for the Attention module.
        """
        super().__init__()
        assert not (
            not row_attn and not col_attn
        ), "row or column attention must be turned on"

        self.row_attn = row_attn
        self.col_attn = col_attn
        self.global_query_attn = global_query_attn

        self.norm = nn.LayerNorm(dim)

        self.attn = Attention(dim=dim, heads=heads, **kwargs)

        self.edges_to_attn_bias = (
            nn.Sequential(
                nn.Linear(dim, heads, bias=False),
                Rearrange("b i j h -> b h i j"),
            )
            if accept_edges
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        edges: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Axial Attention module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, height, width, dim).
            edges (torch.Tensor, optional): The edges tensor for attention bias. Defaults to None.
            mask (torch.Tensor, optional): The mask tensor for masking attention. Defaults to None.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, height, width, dim).
        """
        assert (
            self.row_attn ^ self.col_attn
        ), "has to be either row or column attention, but not both"

        b, h, w, d = x.shape

        x = self.norm(x)

        # axial attention

        if self.col_attn:
            axial_dim = w
            mask_fold_axial_eq = "b h w -> (b w) h"
            input_fold_eq = "b h w d -> (b w) h d"
            output_fold_eq = "(b w) h d -> b h w d"

        elif self.row_attn:
            axial_dim = h
            mask_fold_axial_eq = "b h w -> (b h) w"
            input_fold_eq = "b h w d -> (b h) w d"
            output_fold_eq = "(b h) w d -> b h w d"

        x = rearrange(x, input_fold_eq)

        if exists(mask):
            mask = rearrange(mask, mask_fold_axial_eq)

        attn_bias = None
        if exists(self.edges_to_attn_bias) and exists(edges):
            attn_bias = self.edges_to_attn_bias(edges)
            attn_bias = repeat(
                attn_bias, "b h i j -> (b x) h i j", x=axial_dim
            )

        tie_dim = axial_dim if self.global_query_attn else None

        out = self.attn(
            x, mask=mask, attn_bias=attn_bias, tie_dim=tie_dim
        )
        out = rearrange(out, output_fold_eq, h=h, w=w)

        return out



class MsaAttentionBlock(nn.Module):
    def __init__(self, dim, seq_len, heads, dim_head, dropout=0.0):
        super().__init__()
        self.row_attn = AxialAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            row_attn=True,
            col_attn=False,
            accept_edges=True,
        )
        self.col_attn = AxialAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            row_attn=False,
            col_attn=True,
        )

    def forward(self, x, mask=None, pairwise_repr=None):
        x = self.row_attn(x, mask=mask, edges=pairwise_repr) + x
        x = self.col_attn(x, mask=mask) + x
        return x

# Main model that uses the trainable tokenizer, and attention-based mechanism
class MSAModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embed_dim, num_heads, num_layers):
        super().__init__()

        # Trainable embedding layer for tokenization
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=token_to_idx["PAD"])
        self.MsaAtt = MsaAttentionBlock(dim=128, seq_len=512, heads=8, dim_head=64, dropout=0.3)        

        # Final prediction layer (e.g., for structural prediction)
        self.fc = nn.Linear(embed_dim * max_seq_len, 1)  # Output size depends on prediction task (e.g., contact map, binding score)

    def forward(self, sequences):
        # Token embedding (turn sequences into embeddings)
        embedded = self.embedding(sequences)

        output = self.MsaAtt(embedded)
        # Take information only from the original sequence
        output = output[:,0,:,:]
        output = output.view(self.batch_size,-1)
        # Final prediction
        prediction = self.fc(output)
        return prediction
  
