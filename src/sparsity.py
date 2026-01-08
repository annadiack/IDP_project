import torch
import torch.nn as nn
from typing import Dict


class MaskedLinear(nn.Linear):
    """
    Linear layer with a fixed binary mask applied to the weights.
    Masked weights remain zero throughout training.
    """
    def __init__(self, in_features, out_features, bias=True, sparsity=0.0):
        super().__init__(in_features, out_features, bias)

        # Create binary mask
        self.register_buffer("mask", self._create_mask(self.weight, sparsity))

        # Apply mask once at initialization
        with torch.no_grad():
            self.weight.mul_(self.mask)

    @staticmethod
    def _create_mask(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
        """
        Create a binary mask with given sparsity level.
        sparsity = fraction of weights set to zero.
        """
        num_weights = weight.numel()
        num_keep = int((1.0 - sparsity) * num_weights)

        # Flatten indices and randomly select weights to keep
        perm = torch.randperm(num_weights)
        mask = torch.zeros(num_weights, device=weight.device)
        mask[perm[:num_keep]] = 1.0

        return mask.view_as(weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(input, self.weight * self.mask, self.bias)
