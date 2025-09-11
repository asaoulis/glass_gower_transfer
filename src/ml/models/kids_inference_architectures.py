import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

# Maximum input spatial size (H, W) expected by the CNN to size its head.
# The FlexibleO3 backbone will run a dummy pass with this size to infer head dims.
MAX_INPUT_HW: Tuple[int, int] = (256, 256)


class KidsO3NorthSouthEmbedding(nn.Module):
    """
    Embedding network that:
      - expects a data dict with keys: 'E_south', 'B_south', 'E_north', 'B_north'
      - each entry is a tensor of shape [B, C, H, W] with C=6
      - stacks E and B channelwise per hemisphere to form 12-channel inputs
      - applies the SAME flexible o3 CNN (shared weights) to north and south stacks separately
      - zero-pads the south stack to match the north spatial size (if smaller)
      - concatenates the two feature vectors and passes through a shallow MLP
        to produce a final latent vector of size `latent_dim`.
    """

    def __init__(self, latent_dim: int, cnn_out_dim: int = 256, hidden: int = 12, channels_per_map: int = 6):
        super().__init__()
        self.latent_dim = latent_dim
        self.cnn_out_dim = cnn_out_dim
        self.channels_per_map = channels_per_map

        from .compressors import flexible_o3_model

        in_channels = 2 * channels_per_map  # E + B stacked per hemisphere (6 + 6 = 12)
        # Shared CNN used for both north and south
        self.shared_cnn = flexible_o3_model(
            num_outputs=cnn_out_dim,
            hidden=hidden,
            channels=in_channels,
            max_hw=MAX_INPUT_HW,
            predict_sigmas=False,
        )

        # Shallow head after concatenation of north/south embeddings
        hidden_head = max(latent_dim, 2 * cnn_out_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(2 * cnn_out_dim, hidden_head),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_head, latent_dim),
        )

    @staticmethod
    def _pad_to_size(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        """Zero-pad x on bottom/right to target spatial size (H, W)."""
        _, _, h, w = x.shape
        th, tw = target_hw
        pad_h = max(th - h, 0)
        pad_w = max(tw - w, 0)
        if pad_h == 0 and pad_w == 0:
            return x
        # Pad format: (left, right, top, bottom)
        return F.pad(x, (0, pad_w, 0, pad_h))

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Expect tensors of shape [B, C=6, H, W]
        e_south = data["E_south"]
        b_south = data["B_south"]
        e_north = data["E_north"]
        b_north = data["B_north"]

        # Stack channelwise to get 12-channel inputs per hemisphere
        south_stack = torch.cat([e_south, b_south], dim=1)
        north_stack = torch.cat([e_north, b_north], dim=1)

        # Ensure south matches north spatial size by zero-padding (bottom/right)
        _, _, hn, wn = north_stack.shape
        south_stack = self._pad_to_size(south_stack, (hn, wn))

        # Shared CNN processes both stacks
        south_feat = self.shared_cnn(south_stack)
        north_feat = self.shared_cnn(north_stack)

        # Concatenate features and map to latent representation
        feats = torch.cat([south_feat, north_feat], dim=1)
        z = self.head(feats)
        return z


# Simple registry to integrate with the existing model selection flow
KIDS_MODEL_BUILDERS = {
    "kids_o3_dual": lambda num_outputs, **kwargs: KidsO3NorthSouthEmbedding(latent_dim=num_outputs, **kwargs),
}
