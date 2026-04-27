"""
Self-Modifying Encoder — Nested Learning wrapper.

Differences vs. the draft in the user's message:
  * Inner loop uses `torch.func.functional_call` with an isolated param dict
    instead of `param.data.sub_()` in-place mutation. This avoids silent
    gradient corruption when the outer graph references modifier weights.
  * `self.modifier.parameters()` are NEVER mutated in-place; they are trained
    only through the `aux_surprise` path of the outer optimizer. Self-adaptation
    lives in the ephemeral `adapted_params` dict, which is the correct TTT /
    MAML-style semantics.
  * `SelfModifyingEncoder.forward(x)` returns `(c2, c3, c4, c5)` to match the
    base encoder contract. `forward(x, return_nested_info=True)` returns the
    extra info dict. This keeps integration with FPNDecoder trivial.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch.func import functional_call
except ImportError:  # PyTorch < 2.0
    from torch.nn.utils.stateless import functional_call  # type: ignore


class SurpriseObjective(nn.Module):
    """Local surprise signal: masked reconstruction + feature-stat consistency."""

    def __init__(self, channels: int, spatial_mask_ratio: float = 0.25):
        super().__init__()
        self.channels = channels
        self.spatial_mask_ratio = float(spatial_mask_ratio)

        # 3x3 convs so masked positions see unmasked neighbours.  With 1x1 convs
        # (bias=False), reconstructed at a masked pixel would be 0 regardless of
        # the weights and the gradient to the reconstructor vanishes.
        bottleneck = max(channels // 4, 8)
        self.reconstructor = nn.Sequential(
            nn.Conv2d(channels, bottleneck, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(bottleneck, channels, kernel_size=3, padding=1, bias=False),
        )

        self.register_buffer("running_mean", torch.zeros(1, channels, 1, 1))
        self.register_buffer("running_var", torch.ones(1, channels, 1, 1))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.ema_momentum = 0.1

    def _spatial_mask(self, x: Tensor) -> Tensor:
        # Vectorised random spatial mask (beats the Python per-batch loop).
        B, _, H, W = x.shape
        num_patches = H * W
        num_keep = num_patches - int(num_patches * self.spatial_mask_ratio)
        noise = torch.rand(B, num_patches, device=x.device)
        ids = torch.argsort(noise, dim=1)
        keep_ids = ids[:, :num_keep]
        mask_flat = torch.zeros(B, num_patches, device=x.device, dtype=x.dtype)
        mask_flat.scatter_(1, keep_ids, 1.0)
        return mask_flat.view(B, 1, H, W)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        _, C, _, _ = x.shape
        mask = self._spatial_mask(x)
        masked_x = x * mask
        reconstructed = self.reconstructor(masked_x)
        inv_mask = 1.0 - mask
        denom = inv_mask.sum() * C + 1e-8
        recon_loss = (((reconstructed - x) ** 2) * inv_mask).sum() / denom

        current_mean = x.mean(dim=[0, 2, 3], keepdim=True)
        current_var = x.var(dim=[0, 2, 3], keepdim=True)
        if self.training:
            with torch.no_grad():
                self.num_batches_tracked += 1
                self.running_mean.lerp_(current_mean.detach(), self.ema_momentum)
                self.running_var.lerp_(current_var.detach(), self.ema_momentum)

        consist_loss = (
            (current_mean - self.running_mean).pow(2).mean()
            + (current_var / (self.running_var + 1e-8) - 1).pow(2).mean()
        )
        total = recon_loss + 0.1 * consist_loss
        info = {
            "recon_loss": recon_loss.detach(),
            "consist_loss": consist_loss.detach(),
            "total_surprise": total.detach(),
        }
        return total, info


class SelfModifyingBlock(nn.Module):
    """Nested-Learning self-modifying block with clean autograd.

    At each forward pass we:
      1. Snapshot modifier weights into a dict (`adapted`) that is DETACHED
         from `self.modifier.parameters()`.
      2. Run `inner_steps - 1` TTT-style gradient updates on `adapted` using
         `functional_call` (no in-place mutation of `self.modifier`).
      3. Compute `modification = functional_call(modifier, adapted, features)`.
         Outer gradient to upstream (backbone) flows through `features`; grad
         to `adapted` is a dead-end.
      4. Separately compute `aux_surprise` using the ORIGINAL `self.modifier`
         params — this is the path the outer optimiser uses to learn the
         surprise representation.
    """

    def __init__(
        self,
        channels: int,
        inner_steps: int = 3,
        inner_lr: float = 0.01,
        modifier_expansion: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.inner_steps = max(int(inner_steps), 1)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // 4, 8)),
            nn.GELU(),
            nn.Linear(max(channels // 4, 8), 1),
            nn.Sigmoid(),
        )

        hidden = channels * int(modifier_expansion)
        num_groups = max(min(8, hidden // 4), 1)
        self.modifier = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(num_groups, hidden),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        self.surprise = SurpriseObjective(channels)

        self.residual_scale = nn.Parameter(torch.tensor(0.05))
        self.register_buffer("log_inner_lr", torch.tensor(math.log(float(inner_lr))))

    def _inner_lr(self) -> float:
        return float(self.log_inner_lr.exp().item())

    def _inner_loop(self, features: Tensor) -> Tuple[Tensor, Tensor, List[Dict[str, Tensor]]]:
        inner_lr = self._inner_lr()

        # Start adapted params from current modifier params, but detached so that
        # mutations here never bleed into self.modifier.parameters().
        adapted: Dict[str, Tensor] = {
            name: p.detach().clone().requires_grad_(True)
            for name, p in self.modifier.named_parameters()
        }

        all_info: List[Dict[str, Tensor]] = []
        feat_detached = features.detach()

        with torch.enable_grad():
            for step in range(self.inner_steps):
                modified = functional_call(self.modifier, adapted, (feat_detached,))
                surprise_loss, info = self.surprise(modified + feat_detached)
                info = dict(info)
                info["inner_step"] = step
                all_info.append(info)

                if step < self.inner_steps - 1:
                    params_list = list(adapted.values())
                    grads = torch.autograd.grad(
                        surprise_loss,
                        params_list,
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=True,
                    )
                    new_adapted: Dict[str, Tensor] = {}
                    for (name, p), g in zip(adapted.items(), grads):
                        if g is None:
                            new_adapted[name] = p
                        else:
                            new_adapted[name] = (p - inner_lr * g).detach().requires_grad_(True)
                    adapted = new_adapted

        # Final self-modified output — features keep outer grad (→ backbone);
        # adapted params are detached, so the modifier module itself is unaffected
        # by the seg-loss backward through this path.
        final_modified = functional_call(self.modifier, adapted, (features,))

        # Aux surprise that trains self.modifier + self.surprise via outer loss.
        aux_surprise = features.new_zeros(())
        if any(p.requires_grad for p in self.modifier.parameters()):
            orig_modified = self.modifier(features)
            aux_surprise, _ = self.surprise(orig_modified + features)

        return final_modified, aux_surprise, all_info

    def forward(
        self, x: Tensor, return_info: bool = False
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        modification, aux_surprise, all_info = self._inner_loop(x)

        gate_value = self.gate(x)  # [B, 1]
        gate_map = gate_value.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        output = x + self.residual_scale * gate_map * modification

        info: Optional[Dict[str, Tensor]] = None
        if return_info:
            info = {
                # keep the non-detached gate_value so the outer gate-diversity
                # regulariser can backprop into the gate network
                "gate_value": gate_value.mean(),
                "residual_scale": self.residual_scale.detach(),
                "inner_lr": torch.tensor(self._inner_lr(), device=x.device),
                "inner_steps": all_info,
                "aux_surprise": aux_surprise,
            }
        return output, info


class SelfModifyingEncoder(nn.Module):
    """Wrap a 4-stage encoder with per-stage SelfModifyingBlocks.

    * `forward(x)` → `(c2, c3, c4, c5)` (same contract as the raw encoder).
    * `forward(x, return_nested_info=True)` → `((c2, c3, c4, c5), nested_info)`.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_channels: List[int],
        inner_steps_schedule: List[int] = (1, 2, 3, 4),
        inner_lr: float = 0.01,
        apply_stages: Optional[List[int]] = None,
        modifier_expansion: int = 2,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if apply_stages is None:
            apply_stages = [2, 3]
        self.apply_stages = [int(s) for s in apply_stages]
        if any(s < 0 or s >= len(feature_channels) for s in self.apply_stages):
            raise ValueError(f"apply_stages {self.apply_stages} out of range for {len(feature_channels)} stages")

        self.modifiers = nn.ModuleDict()
        self.pre_norms = nn.ModuleDict()
        for stage_idx in self.apply_stages:
            ch = int(feature_channels[stage_idx])
            steps = int(inner_steps_schedule[stage_idx]) if stage_idx < len(inner_steps_schedule) else 1
            stage_lr = float(inner_lr) * (0.5 ** stage_idx)
            key = f"stage_{stage_idx}"
            self.modifiers[key] = SelfModifyingBlock(
                channels=ch,
                inner_steps=steps,
                inner_lr=stage_lr,
                modifier_expansion=modifier_expansion,
                dropout=dropout,
            )
            self.pre_norms[key] = nn.GroupNorm(max(min(8, ch // 4), 1), ch)

        # Expose base encoder channels for FPN-compat.
        self.out_channels = list(getattr(backbone, "out_channels", feature_channels))
        self.pretrained_loaded = bool(getattr(backbone, "pretrained_loaded", False))
        self.checkpoint_compatible = bool(getattr(backbone, "checkpoint_compatible", True))

    def forward(
        self, x: Tensor, return_nested_info: bool = False
    ):
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        features = list(features)
        nested_info: Dict[str, Dict[str, Tensor]] = {}

        for stage_idx in self.apply_stages:
            key = f"stage_{stage_idx}"
            feat = self.pre_norms[key](features[stage_idx])
            modified, info = self.modifiers[key](feat, return_info=return_nested_info)
            features[stage_idx] = modified
            if return_nested_info and info is not None:
                nested_info[key] = info

        features = tuple(features)
        if return_nested_info:
            return features, nested_info
        return features
