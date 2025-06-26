import torch
import torch.nn as nn

class DyM(nn.Module):
    """
    Dynamic Mask: applies to all tokens (including CLS), 
    initialized to identity (mask=1), logs basic statistics each forward.
    """
    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: dimensionality of each token embedding.
        """
        super().__init__()
        # Linear mapping from each embedding vector to a single logit
        self.mask_layer = nn.Linear(embed_dim, 1, bias=True)
        # Initialize so that 2 * sigmoid(0) == 1.0 exactly
        nn.init.zeros_(self.mask_layer.weight)
        nn.init.zeros_(self.mask_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, L, D)
               where B=batch size, L=sequence length, D=embed_dim.
        Returns:
            gated: Tensor of same shape with each token scaled by its mask.
        """
        # Compute per-token logits -> mask in [0, 2]
        logits = self.mask_layer(x)               # shape (B, L, 1)
        mask   = 2 * torch.sigmoid(logits)        # shape (B, L, 1)

#        # Debug statistics
#        with torch.no_grad():
#            flat = mask.view(-1)
#            mn, mx = flat.min().item(), flat.max().item()
#            mean, std = flat.mean().item(), flat.std().item()
#            print(f"[DyM] mask stats → min:{mn:.3f}, mean:{mean:.3f}, std:{std:.3f}, max:{mx:.3f}")

        # Apply mask to all tokens
        gated = x * mask                           # broadcasting over D

        return gated
