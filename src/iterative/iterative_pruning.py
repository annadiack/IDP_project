import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


# -------------------------
# Prunable layers (masked)
# -------------------------

class PrunableLinear(nn.Linear):
    """Linear with a persistent binary mask buffer (same shape as weight)."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones_like(self.weight))

    @torch.no_grad()
    def apply_mask_(self):
        self.weight.mul_(self.mask)

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.mask, self.bias)


class PrunableConv2d(nn.Conv2d):
    """Conv2d with a persistent binary mask buffer (same shape as weight)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.ones_like(self.weight))

    @torch.no_grad()
    def apply_mask_(self):
        self.weight.mul_(self.mask)

    def forward(self, x):
        return nn.functional.conv2d(
            x,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# -------------------------
# Utilities
# -------------------------

def iter_prunable_params(model: nn.Module) -> Iterable[Tuple[str, torch.Tensor, torch.Tensor]]:
    """Yield (module_name, weight, mask) for modules that have both weight and mask."""
    for name, m in model.named_modules():
        if hasattr(m, "weight") and hasattr(m, "mask"):
            w = getattr(m, "weight")
            mask = getattr(m, "mask")
            if isinstance(w, torch.Tensor) and isinstance(mask, torch.Tensor):
                yield name, w, mask


@torch.no_grad()
def apply_all_masks_(model: nn.Module) -> None:
    """Force weights to respect masks (hard zero)."""
    for _, m in model.named_modules():
        if hasattr(m, "apply_mask_"):
            m.apply_mask_()


@torch.no_grad()
def global_sparsity(model: nn.Module) -> float:
    """Fraction of zeros implied by masks across all prunable tensors."""
    total = 0
    nnz = 0
    for _, _, mask in iter_prunable_params(model):
        total += mask.numel()
        nnz += int(mask.sum().item())
    if total == 0:
        return 0.0
    return 1.0 - (nnz / total)


@torch.no_grad()
def global_magnitude_prune_(
    model: nn.Module,
    prune_fraction_of_remaining: float,
    include_bias: bool = False,
) -> Dict[str, float]:
    """
    Global unstructured magnitude pruning over ALL prunable weights.

    prune_fraction_of_remaining:
      fraction of currently-unpruned weights to prune at this step.

    include_bias:
      typically False; pruning biases is uncommon.
    """
    if not (0.0 <= prune_fraction_of_remaining < 1.0):
        raise ValueError("prune_fraction_of_remaining must be in [0, 1).")

    # Collect magnitudes of ACTIVE weights only
    mags: List[torch.Tensor] = []
    meta: List[Tuple[torch.Tensor, torch.Tensor]] = []  # (weight, mask)

    for _, w, mask in iter_prunable_params(model):
        active = mask.bool()
        if active.any():
            mags.append(w.detach().abs()[active].flatten())
            meta.append((w, mask))

    if not mags:
        return {"pruned_now": 0.0, "sparsity": global_sparsity(model)}

    all_mags = torch.cat(mags)
    active_total = all_mags.numel()

    k = int(math.floor(prune_fraction_of_remaining * active_total))
    if k <= 0:
        return {"pruned_now": 0.0, "sparsity": global_sparsity(model)}

    # kth smallest magnitude among active weights -> threshold
    thresh = torch.kthvalue(all_mags.cpu(), k).values.item()

    pruned_count = 0
    for w, mask in meta:
        active = mask.bool()
        to_prune = active & (w.detach().abs() <= thresh)
        mask[to_prune] = 0.0
        pruned_count += int(to_prune.sum().item())

    apply_all_masks_(model)

    pruned_now = 0.0 if active_total == 0 else (pruned_count / active_total)
    return {"pruned_now": float(pruned_now), "sparsity": float(global_sparsity(model))}


@dataclass
class IterativePruneSchedule:
    """
    Choose a constant fraction-per-step of remaining weights so that
    after `steps` pruning events, you reach `final_sparsity` (approximately).

    Density after steps: (1 - f)^steps
    Target density: 1 - final_sparsity
    => f = 1 - (1 - final_sparsity)^(1/steps)
    """
    final_sparsity: float
    steps: int

    def __post_init__(self):
        if not (0.0 <= self.final_sparsity < 1.0):
            raise ValueError("final_sparsity must be in [0, 1).")
        if self.steps < 1:
            raise ValueError("steps must be >= 1.")

    def fraction_per_step(self) -> float:
        target_density = 1.0 - self.final_sparsity
        f = 1.0 - (target_density ** (1.0 / self.steps))
        return float(max(0.0, min(0.999999, f)))
