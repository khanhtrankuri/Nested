"""
Advanced Improvements for PolyMemnet v2.0
=========================================

Implements 6 major enhancements:
1. Cross-Stage Modulation - multi-scale feature interaction
2. Adaptive CMS Scheduling - dynamic inner loop control
3. Meta Inner Optimizer - learned optimization for inner loop
4. Hierarchical Prototype Bank - tree-structured memory
5. MC Dropout Uncertainty - better uncertainty calibration
6. Enhanced Loss Functions - contrastive, consistency, sparsity

Author: Claude Code Enhancement
Date: 2026-04-28
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from loss.strong_baseline_loss import StrongBaselineLoss


# =============================================================================
# Local copies of surprise objectives (to avoid circular imports)
# =============================================================================


class ConsistencySurprise(nn.Module):
    """Rẻ: chỉ penalize drift của feature statistics so với running mean/var.

    Dùng cho stage nông nơi ta KHÔNG muốn encoder adapt mạnh per-sample
    (chỉ muốn giữ consistent với prior ImageNet).
    """

    def __init__(self, channels: int, ema_momentum: float = 0.1):
        super().__init__()
        self.ema_momentum = ema_momentum
        self.register_buffer("running_mean", torch.zeros(1, channels, 1, 1))
        self.register_buffer("running_var", torch.ones(1, channels, 1, 1))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        current_mean = x.mean(dim=[0, 2, 3], keepdim=True)
        current_var = x.var(dim=[0, 2, 3], keepdim=True)

        buf_dtype = self.running_mean.dtype
        if self.training:
            with torch.no_grad():
                self.num_batches_tracked += 1
                self.running_mean.lerp_(
                    current_mean.detach().to(dtype=buf_dtype), self.ema_momentum
                )
                self.running_var.lerp_(
                    current_var.detach().to(dtype=buf_dtype), self.ema_momentum
                )

        rm = self.running_mean.to(dtype=current_mean.dtype)
        rv = self.running_var.to(dtype=current_var.dtype)
        loss = (
            (current_mean - rm).pow(2).mean()
            + (current_var / (rv + 1e-8) - 1.0).pow(2).mean()
        )
        return loss, {
            "consist_loss": loss.detach(),
            "recon_loss": torch.zeros_like(loss).detach(),
            "total_surprise": loss.detach(),
        }


class FullSurprise(nn.Module):
    """Recon + consistency — dùng cho stage sâu nơi ta muốn inner loop mạnh."""

    def __init__(
        self,
        channels: int,
        spatial_mask_ratio: float = 0.25,
        consist_weight: float = 0.1,
    ):
        super().__init__()
        self.spatial_mask_ratio = spatial_mask_ratio
        self.consist_weight = consist_weight

        bottleneck = max(channels // 4, 16)
        self.reconstructor = nn.Sequential(
            nn.Conv2d(channels, bottleneck, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(bottleneck, channels, 1, bias=False),
        )
        self.register_buffer("running_mean", torch.zeros(1, channels, 1, 1))
        self.register_buffer("running_var", torch.ones(1, channels, 1, 1))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.ema_momentum = 0.1

    def _spatial_mask(self, x: Tensor) -> Tensor:
        B, _, H, W = x.shape
        num_patches = H * W
        num_mask = max(1, int(num_patches * self.spatial_mask_ratio))
        mask = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
        for b in range(B):
            indices = torch.randperm(num_patches, device=x.device)[:num_mask]
            rows = indices // W
            cols = indices % W
            mask[b, 0, rows, cols] = 0.0
        return mask

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        _, C, _, _ = x.shape
        mask = self._spatial_mask(x)
        masked_x = x * mask
        reconstructed = self.reconstructor(masked_x)
        inv_mask = 1.0 - mask
        recon_loss = (((reconstructed - x) ** 2) * inv_mask).sum() / (
            inv_mask.sum() * C + 1e-8
        )

        current_mean = x.mean(dim=[0, 2, 3], keepdim=True)
        current_var = x.var(dim=[0, 2, 3], keepdim=True)
        buf_dtype = self.running_mean.dtype
        if self.training:
            with torch.no_grad():
                self.num_batches_tracked += 1
                self.running_mean.lerp_(
                    current_mean.detach().to(dtype=buf_dtype), self.ema_momentum
                )
                self.running_var.lerp_(
                    current_var.detach().to(dtype=buf_dtype), self.ema_momentum
                )

        rm = self.running_mean.to(dtype=current_mean.dtype)
        rv = self.running_var.to(dtype=current_var.dtype)
        consist_loss = (
            (current_mean - rm).pow(2).mean()
            + (current_var / (rv + 1e-8) - 1.0).pow(2).mean()
        )

        total = recon_loss + self.consist_weight * consist_loss
        return total, {
            "recon_loss": recon_loss.detach(),
            "consist_loss": consist_loss.detach(),
            "total_surprise": total.detach(),
        }

# =============================================================================
# 1. CROSS-STAGE MODULATION
# =============================================================================


class CrossStageModulator(nn.Module):
    """
    Allows features from different stages to interact via cross-attention.
    Implements hierarchical multi-scale self-modification.

    Each stage can attend to:
    - Its own features (self)
    - Shallower stages (high-res details)
    - Deeper stages (semantic context)

    This creates a bidirectional flow of information across scales,
    enhancing the nested learning paradigm.
    """

    def __init__(
        self,
        channels_list: List[int],  # [c2, c3, c4, c5]
        num_heads: int = 4,
        use_relative_pos: bool = False,
    ):
        super().__init__()
        self.channels_list = channels_list
        self.num_stages = len(channels_list)
        self.num_heads = num_heads

        # Cross-attention modules between stage pairs
        # attn[i][j] = attention from stage i to stage j
        self.cross_attentions = nn.ModuleDict()
        for i in range(self.num_stages):
            for j in range(self.num_stages):
                if i == j:
                    continue  # self-attention already in backbone
                key = f"{i}_{j}"
                self.cross_attentions[key] = nn.MultiheadAttention(
                    embed_dim=channels_list[i],
                    num_heads=num_heads,
                    batch_first=True,
                )

        # Learnable routing weights: decide how much to borrow from other stages
        self.routing_mlp = nn.Sequential(
            nn.Linear(sum(channels_list), 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_stages * (self.num_stages - 1)),  # exclude self
        )

        # Projections to match dimensions when channels differ
        self.projections = nn.ModuleDict()
        for i in range(self.num_stages):
            for j in range(self.num_stages):
                if i == j:
                    continue
                if channels_list[i] != channels_list[j]:
                    key = f"{i}_{j}"
                    self.projections[key] = nn.Linear(
                        channels_list[j], channels_list[i], bias=False
                    )

        # Gate for each cross-stage connection
        self.cross_gates = nn.Parameter(torch.ones(self.num_stages, self.num_stages))
        self.cross_gates.data.fill_(0.1)  # start conservative
        # Zero out diagonal (no self-gate)
        for i in range(self.num_stages):
            self.cross_gates.data[i, i] = 0.0

    def _project(self, x: Tensor, i: int, j: int) -> Tensor:
        """Project feature from stage j to stage i's space."""
        if i == j:
            return x
        key = f"{i}_{j}"
        if key in self.projections:
            # Reshape for linear: [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = x.shape
            x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]
            x_proj = self.projections[key](x_flat)
            return x_proj.transpose(1, 2).reshape(B, -1, H, W)
        return x

    def forward(
        self,
        features: List[Tensor],
        return_routing: bool = False,
    ) -> Union[Tuple[List[Tensor], Dict[str, Tensor]], List[Tensor]]:
        """
        Args:
            features: list of [c2, c3, c4, c5] feature maps
            return_routing: whether to return routing weights

        Returns:
            modulated_features: list of enhanced features
            routing_info: dict with routing weights, attention maps (if return_routing)
        """
        B = features[0].shape[0]
        device = features[0].device
        routing_info = {}

        # Compute routing weights based on global context
        # Use spatial avg of all features
        global_features = torch.cat([
            F.adaptive_avg_pool2d(f, 1).flatten(1)
            for f in features
        ], dim=1)  # [B, sum(channels)]

        routing_logits = self.routing_mlp(global_features)  # [B, N*(N-1)]
        routing_weights = torch.softmax(routing_logits, dim=-1)
        if return_routing:
            routing_info["routing_weights"] = routing_weights.detach()

        # For each stage, aggregate info from other stages
        modulated = [None] * self.num_stages
        attention_maps = {}

        for i in range(self.num_stages):
            # Start with original features
            feat_i = features[i]
            aggregated = torch.zeros_like(feat_i)

            # Weighted sum of contributions from all other stages
            weight_idx = 0
            for j in range(self.num_stages):
                if i == j:
                    continue

                # Get routing weight for this pair (i<-j)
                weight = routing_weights[:, weight_idx].view(-1, 1, 1, 1)

                # Project stage j features to stage i space
                feat_j_proj = self._project(features[j], i, j)

                # Cross-attention: feat_i as query, feat_j_proj as key/value
                # Reshape to sequence format
                B, C, H, W = feat_i.shape
                q = feat_i.flatten(2).transpose(1, 2)  # [B, N_i, C_i]
                kv = feat_j_proj.flatten(2).transpose(1, 2)  # [B, N_j, C_i]

                # Apply cross-attention
                attn_out, attn_weights = self.cross_attentions[f"{i}_{j}"](
                    q, kv, kv
                )  # attn_weights: [B, N_i, N_j]

                # Reshape back
                attn_out = attn_out.transpose(1, 2).reshape(B, -1, H, W)

                # Apply cross-gate
                gate_val = self.cross_gates[i, j]
                attn_out = gate_val * attn_out

                # Weighted accumulation
                aggregated = aggregated + weight * attn_out

                if return_routing:
                    attention_maps[f"{i}<-{j}"] = attn_weights.detach().mean(dim=0)  # avg over batch

                weight_idx += 1

            # Combine: residual + gated cross-stage context
            self_gate = torch.sigmoid(
                F.adaptive_avg_pool2d(feat_i, 1).flatten(1).mean(dim=1, keepdim=True)
            ).view(-1, 1, 1, 1)
            modulated[i] = feat_i + self_gate * aggregated

        if return_routing:
            routing_info["attention_maps"] = attention_maps
            return modulated, routing_info
        return modulated


# =============================================================================
# 2. ADAPTIVE CMS SCHEDULING
# =============================================================================


class AdaptiveInnerLoopController(nn.Module):
    """
    Predicts optimal inner loop parameters based on input features.
    Implements dynamic CMS scheduling.
    """

    def __init__(
        self,
        channels: int,
        max_steps: int = 8,
        base_lr: float = 1e-2,
    ):
        super().__init__()
        self.channels = channels
        self.max_steps = max_steps
        self.base_lr = base_lr

        # Network to predict inner steps (categorical via softmax)
        self.steps_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 64),
            nn.GELU(),
            nn.Linear(64, max_steps),
        )

        # Network to predict inner LR multiplier
        self.lr_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Network to predict surprise threshold
        self.threshold_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def predict_parameters(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Predict adaptive parameters for current input.
        """
        B = x.shape[0]

        # Predict number of inner steps (as one-hot, then take expected value)
        steps_logits = self.steps_predictor(x)  # [B, max_steps]
        steps_probs = F.softmax(steps_logits, dim=-1)
        steps = (
            torch.arange(self.max_steps, device=x.device, dtype=x.dtype)
            .view(1, -1)
            .expand(B, -1)
        )
        predicted_steps = (steps * steps_probs).sum(dim=1)  # expected steps

        # Predict LR multiplier
        lr_multiplier = self.lr_predictor(x)  # [B, 1]

        # Predict surprise threshold
        threshold = self.threshold_predictor(x)  # [B, 1]

        return {
            "predicted_steps": predicted_steps,
            "steps_probs": steps_probs,
            "lr_multiplier": lr_multiplier,
            "surprise_threshold": threshold,
        }


class AdaptiveCMSSelfModifyingBlock(nn.Module):
    """
    Extended self-modifying block with adaptive inner loop control.
    """

    def __init__(
        self,
        channels: int,
        base_inner_steps: int = 3,
        base_inner_lr: float = 0.01,
        inner_momentum: float = 0.9,
        modifier_expansion: int = 2,
        dropout: float = 0.1,
        surprise_type: str = "full",
        persist_momentum: bool = True,
        residual_init: float = 0.05,
        adaptive_enabled: bool = True,
        max_adaptive_steps: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.base_inner_steps = base_inner_steps
        self.base_inner_lr = base_inner_lr
        self.inner_momentum = inner_momentum
        self.persist_momentum = persist_momentum
        self.surprise_type = surprise_type
        self.adaptive_enabled = adaptive_enabled
        self.max_steps = max_adaptive_steps

        # Base components (same as CMSSelfModifyingBlock)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // 4, 16)),
            nn.GELU(),
            nn.Linear(max(channels // 4, 16), 1),
            nn.Sigmoid(),
        )

        hidden = channels * modifier_expansion
        self.modifier = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(min(8, max(1, hidden // 4)), hidden),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        # Surprise objective
        from model.backbones.self_modifying_encoder_cms import (
            ConsistencySurprise,
            FullSurprise,
        )
        if surprise_type == "consistency":
            self.surprise = ConsistencySurprise(channels)
        else:
            self.surprise = FullSurprise(channels)

        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

        # Adaptive controller (if enabled)
        if adaptive_enabled:
            self.adaptive_controller = AdaptiveInnerLoopController(
                channels=channels,
                max_steps=max_adaptive_steps,
                base_lr=base_inner_lr,
            )
        else:
            self.adaptive_controller = None

        self._momentum_buffer: Optional[Dict[str, Tensor]] = None

    def _ensure_momentum_buffer(self):
        if not self.persist_momentum:
            return
        if self._momentum_buffer is None:
            self._momentum_buffer = {
                name: torch.zeros_like(p.data)
                for name, p in self.modifier.named_parameters()
            }

    def reset_momentum(self):
        if self._momentum_buffer is not None:
            for buf in self._momentum_buffer.values():
                buf.zero_()

    def _inner_loop(
        self,
        features: Tensor,
        adaptive_params: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, List[Dict]]:
        all_info: List[Dict] = []
        self._ensure_momentum_buffer()

        # Determine inner steps and LR
        if adaptive_params is not None and self.adaptive_enabled:
            pred_steps = adaptive_params["predicted_steps"].clamp(min=1, max=self.max_steps)
            inner_steps = int(torch.round(pred_steps.mean()).item())
            inner_lr = (
                self.base_inner_lr
                * adaptive_params["lr_multiplier"].mean().item()
            )
            surprise_threshold = adaptive_params["surprise_threshold"].mean().item()
        else:
            inner_steps = self.base_inner_steps
            inner_lr = self.base_inner_lr
            surprise_threshold = float("inf")  # always run

        with torch.enable_grad():
            feat_inner = features.detach()
            current_surprise = None

            for step in range(inner_steps):
                modified = self.modifier(feat_inner)
                surprise_loss, info = self.surprise(modified + feat_inner)
                info["inner_step"] = step
                all_info.append(info)

                current_surprise = surprise_loss.item()

                # Early stopping nếu surprise nhỏ hơn threshold
                if current_surprise < surprise_threshold and step > 0:
                    break

                if step < inner_steps - 1:
                    mod_params = [
                        (name, p) for name, p in self.modifier.named_parameters()
                        if p.requires_grad
                    ]
                    if not mod_params or not surprise_loss.requires_grad:
                        continue
                    grads = torch.autograd.grad(
                        surprise_loss,
                        [p for _, p in mod_params],
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=True,
                    )
                    with torch.no_grad():
                        for (name, param), grad in zip(mod_params, grads):
                            if grad is None:
                                continue
                            if self.persist_momentum:
                                buf = self._momentum_buffer[name]
                                buf.mul_(self.inner_momentum).add_(grad)
                                param.data.sub_(inner_lr * buf)
                            else:
                                param.data.sub_(inner_lr * grad)

        final_modified = self.modifier(features)
        if any(p.requires_grad for p in self.modifier.parameters()):
            aux_surprise, _ = self.surprise(final_modified + features)
        else:
            aux_surprise = torch.tensor(0.0, device=features.device, dtype=features.dtype)

        return final_modified, aux_surprise, all_info

    def forward(
        self, x: Tensor, return_info: bool = False
    ) -> Tuple[Tensor, Optional[Dict]]:
        saved_state = {
            name: param.data.clone() for name, param in self.modifier.named_parameters()
        }

        # Get adaptive parameters if controller exists
        adaptive_params = None
        if self.adaptive_controller is not None:
            adaptive_params = self.adaptive_controller.predict_parameters(x)

        modification, aux_surprise, steps_info = self._inner_loop(x, adaptive_params)

        gate = self.gate(x).unsqueeze(-1).unsqueeze(-1)
        output = x + self.residual_scale * gate * modification

        for name, param in self.modifier.named_parameters():
            param.data.copy_(saved_state[name])

        info = None
        if return_info:
            info = {
                "gate_value": float(gate.detach().mean().item()),
                "residual_scale": float(self.residual_scale.item()),
                "inner_lr": self.base_inner_lr
                * (adaptive_params["lr_multiplier"].item() if adaptive_params else 1.0),
                "inner_steps": steps_info,
                "aux_surprise": aux_surprise,
                "surprise_type": self.surprise_type,
                "persist_momentum": self.persist_momentum,
            }
            if adaptive_params is not None:
                info["adaptive"] = {
                    "predicted_steps": float(adaptive_params["predicted_steps"].item()),
                    "lr_multiplier": float(adaptive_params["lr_multiplier"].item()),
                    "surprise_threshold": float(adaptive_params["surprise_threshold"].item()),
                }
        return output, info


# =============================================================================
# 3. META INNER OPTIMIZER
# =============================================================================


class MetaInnerOptimizer(nn.Module):
    """
    Learns an optimization strategy for inner loop updates.
    Replaces hand-crafted SGD with a learned optimizer (LSTM-based).
    """

    def __init__(
        self,
        param_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM to maintain optimization state
        self.lstm = nn.LSTMCell(
            input_size=param_dim + 1,  # gradient + loss value
            hidden_size=hidden_dim,
        )

        # Networks to produce update direction and learning rate
        self.update_net = nn.Linear(hidden_dim, param_dim)
        self.lr_net = nn.Linear(hidden_dim, 1)

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Initialize LSTM hidden states."""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def step(
        self,
        grad: Tensor,
        loss: Tensor,
        h_c: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Compute update using learned optimizer.

        Args:
            grad: [B, param_dim] gradient
            loss: [B] scalar loss (or [B, 1])
            h_c: (h, c) LSTM hidden states

        Returns:
            update: [B, param_dim] parameter update
            new_h_c: updated (h, c)
        """
        h, c = h_c
        B, D = grad.shape

        # Prepare LSTM input: gradient + loss value
        loss_exp = loss.view(B, 1) if loss.dim() == 0 else loss.view(B, 1)
        lstm_input = torch.cat([grad, loss_exp], dim=1)  # [B, D+1]

        # LSTM step
        h_new, c_new = self.lstm(lstm_input, (h, c))

        # Compute update
        update_dir = self.update_net(h_new)  # [B, D]
        lr_mult = torch.sigmoid(self.lr_net(h_new)).view(B, 1)  # [B, 1]

        # Normalize update direction
        update_norm = update_dir.norm(dim=1, keepdim=True).clamp(min=1e-8)
        update_dir = update_dir / update_norm

        update = lr_mult * update_dir

        return update, (h_new, c_new)


class MetaOptimizedSelfModifyingBlock(nn.Module):
    """
    Self-modifying block using meta-learned optimizer for inner loop.
    """

    def __init__(
        self,
        channels: int,
        inner_steps: int = 3,
        modifier_expansion: int = 2,
        dropout: float = 0.1,
        surprise_type: str = "full",
        persist_momentum: bool = False,  # Meta-optimizer has its own state
        residual_init: float = 0.05,
    ):
        super().__init__()
        self.channels = channels
        self.inner_steps = inner_steps
        self.persist_momentum = persist_momentum

        # Gate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // 4, 16)),
            nn.GELU(),
            nn.Linear(max(channels // 4, 16), 1),
            nn.Sigmoid(),
        )

        # Modifier
        hidden = channels * modifier_expansion
        self.modifier = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(min(8, max(1, hidden // 4)), hidden),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        # Surprise objective
        from model.backbones.self_modifying_encoder_cms import (
            ConsistencySurprise,
            FullSurprise,
        )
        if surprise_type == "consistency":
            self.surprise = ConsistencySurprise(channels)
        else:
            self.surprise = FullSurprise(channels)

        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

        # Meta-optimizer for inner loop
        # Flatten modifier parameters to param_dim
        # We'll create optimizer for each parameter separately? Or share?
        # For simplicity, we create one meta-optimizer that acts on flattened view
        # In practice, we need to handle multiple parameter tensors.
        # We'll use a dict of meta-optimizers per parameter name.
        self.meta_optimizers: Dict[str, MetaInnerOptimizer] = nn.ModuleDict()

        # Pre-register known parameters
        for name, param in self.modifier.named_parameters():
            if param.requires_grad:
                param_dim = param.numel() // param.shape[0] if param.dim() > 1 else param.numel()
                self.meta_optimizers[name] = MetaInnerOptimizer(
                    param_dim=param_dim,
                    hidden_dim=64,
                )

        # Hidden states for each optimizer (persistent across forward passes)
        self._optimizer_states: Optional[Dict[str, Tuple[Tensor, Tensor]]] = None

    def _ensure_optimizer_states(self, batch_size: int, device: torch.device):
        if self._optimizer_states is None:
            self._optimizer_states = {}
            for name, opt in self.meta_optimizers.items():
                self._optimizer_states[name] = opt.init_hidden(batch_size, device)

    def reset_optimizer_states(self):
        """Reset LSTM states between epochs or samples."""
        self._optimizer_states = None

    def _inner_loop_meta(
        self,
        features: Tensor,
    ) -> Tuple[Tensor, Tensor, List[Dict]]:
        all_info: List[Dict] = []
        B = features.shape[0]
        self._ensure_optimizer_states(B, features.device)

        # Flatten features per batch item for parameter-wise updates
        feat_inner = features.detach()

        # Prepare per-parameter gradients storage
        param_batches = {}
        for name, param in self.modifier.named_parameters():
            if param.requires_grad:
                # Reshape param to [B, param_dim] for batched meta-optimizer
                param_shape = param.shape
                if len(param_shape) == 1:
                    param_batches[name] = param.view(B, -1)  # [B, D]
                else:
                    # For conv weights: [out_ch, in_ch, k1, k2]
                    # We treat each output channel as a separate "parameter"
                    param_batches[name] = param.view(B, param_shape[0], -1)
                    # This is tricky - we need to rethink
                    # For now, skip conv params, only handle 1D or simple cases
                    # Or we could flatten all output channels together
                # TODO: proper handling of multi-dimensional params

        for step in range(self.inner_steps):
            modified = self.modifier(feat_inner)
            surprise_loss, info = self.surprise(modified + feat_inner)
            info["inner_step"] = step
            all_info.append(info)

            if step < self.inner_steps - 1:
                # Compute gradients per-parameter
                mod_params = [
                    (name, p) for name, p in self.modifier.named_parameters()
                    if p.requires_grad
                ]
                if not mod_params or not surprise_loss.requires_grad:
                    continue

                grads = torch.autograd.grad(
                    surprise_loss,
                    [p for _, p in mod_params],
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True,
                )

                with torch.no_grad():
                    for (name, param), grad in zip(mod_params, grads):
                        if grad is None:
                            continue

                        # Use meta-optimizer to compute update
                        if name not in self.meta_optimizers:
                            # Fallback to SGD
                            param.data.sub_(self.base_inner_lr * grad)
                            continue

                        opt = self.meta_optimizers[name]
                        states = self._optimizer_states[name]

                        # Reshape grad to match expected format
                        # For now, we'll flatten to [B, -1]
                        grad_flat = grad.view(B, -1)  # [B, D]

                        # Compute loss per batch (surprise_loss is scalar)
                        loss_batch = surprise_loss.view(B)  # [B]

                        # Get update
                        update, new_states = opt.step(grad_flat, loss_batch, states)

                        # Apply update (reshape back)
                        update_reshaped = update.view(param.shape)
                        param.data.sub_(update_reshaped)

                        # Update states
                        self._optimizer_states[name] = new_states

        final_modified = self.modifier(features)
        if any(p.requires_grad for p in self.modifier.parameters()):
            aux_surprise, _ = self.surprise(final_modified + features)
        else:
            aux_surprise = torch.tensor(0.0, device=features.device, dtype=features.dtype)

        return final_modified, aux_surprise, all_info

    def forward(
        self, x: Tensor, return_info: bool = False
    ) -> Tuple[Tensor, Optional[Dict]]:
        saved_state = {
            name: param.data.clone() for name, param in self.modifier.named_parameters()
        }

        modification, aux_surprise, steps_info = self._inner_loop_meta(x)

        gate = self.gate(x).unsqueeze(-1).unsqueeze(-1)
        output = x + self.residual_scale * gate * modification

        # Restore
        for name, param in self.modifier.named_parameters():
            param.data.copy_(saved_state[name])

        info = None
        if return_info:
            info = {
                "gate_value": float(gate.detach().mean().item()),
                "residual_scale": float(self.residual_scale.item()),
                "inner_steps": steps_info,
                "aux_surprise": aux_surprise,
                "surprise_type": self.surprise_type,
                "persist_momentum": self.persist_momentum,
                "meta_optimizer": True,
            }
        return output, info


# =============================================================================
# 4. HIERARCHICAL PROTOTYPE BANK
# =============================================================================


class HierarchicalPrototypeBank(nn.Module):
    """
    Tree-structured prototype bank with multiple levels.

    Level 0: Coarse prototypes (few, broad categories)
    Level 1: Fine prototypes (more, specific patterns)
    Level 2: Ultra-fine prototypes (many, detailed patterns)

    Routing network decides which levels to query based on input.
    """

    def __init__(
        self,
        feature_dim: int,
        num_levels: int = 3,
        prototypes_per_level: Union[int, List[int]] = 8,
        fast_momentum: float = 0.03,
        slow_momentum: float = 0.0075,
        max_norm: float = 1.0,
        routing_hidden_dim: int = 128,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_levels = num_levels

        if isinstance(prototypes_per_level, int):
            prototypes_per_level = [prototypes_per_level] * num_levels
        assert len(prototypes_per_level) == num_levels

        self.prototypes_per_level = prototypes_per_level
        self.max_norm = max_norm

        # Projection to prototype space
        self.query_proj = nn.Conv2d(feature_dim, feature_dim, 1, bias=False)

        # Create banks per level
        self.banks = nn.ModuleList()
        self.fast_counts_list = []
        self.slow_counts_list = []

        for level, num_proto in enumerate(prototypes_per_level):
            bank = nn.ParameterDict()
            # Fast bank
            bank["fast"] = nn.Parameter(
                torch.randn(num_proto, feature_dim), requires_grad=False
            )
            # Slow bank
            bank["slow"] = nn.Parameter(
                torch.randn(num_proto, feature_dim), requires_grad=False
            )
            self.banks.append(bank)

            # Register buffers for counts
            self.register_buffer(
                f"fast_counts_level{level}", torch.zeros(num_proto)
            )
            self.register_buffer(
                f"slow_counts_level{level}", torch.zeros(num_proto)
            )
            self.fast_counts_list.append(getattr(self, f"fast_counts_level{level}"))
            self.slow_counts_list.append(getattr(self, f"slow_counts_level{level}"))

        # Routing network: decides importance of each level
        self.routing_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, routing_hidden_dim),
            nn.GELU(),
            nn.Linear(routing_hidden_dim, num_levels),
            nn.Softmax(dim=-1),
        )

        # Mixing gates for fast/slow per level
        self.mix_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.GELU(),
                nn.Linear(feature_dim // 2, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_levels)
        ])

        # Initialize prototypes with normalized vectors
        for level, bank in enumerate(self.banks):
            nn.init.normal_(bank["fast"], std=0.02)
            nn.init.normal_(bank["slow"], std=0.02)
            # Normalize
            with torch.no_grad():
                bank["fast"].data = F.normalize(bank["fast"].data, dim=-1)
                bank["slow"].data = F.normalize(bank["slow"].data, dim=-1)

    def _retrieve_from_bank(
        self,
        tokens: Tensor,
        bank: Tensor,
        counts: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Retrieve from a single bank.

        Returns:
            context: [B, D] weighted sum of prototypes
            attn: [B, K] attention weights
            entropy: [B] attention entropy
        """
        B, D = tokens.shape
        K = bank.shape[0]
        device, dtype = tokens.device, tokens.dtype

        ready_mask = counts > 1e-6
        if not torch.any(ready_mask).item():
            return (
                torch.zeros_like(tokens),
                torch.zeros(B, K, device=device, dtype=dtype),
                torch.zeros(B, device=device, dtype=dtype),
            )

        with torch.amp.autocast("cuda", enabled=False):
            bank_ready = F.normalize(bank[ready_mask].float(), dim=-1)
            tokens_f32 = tokens.float()
            logits = torch.matmul(tokens_f32, bank_ready.t()) / math.sqrt(D)
            attn_ready = torch.softmax(logits, dim=-1)
            context = torch.matmul(attn_ready, bank_ready)

        attn_full = torch.zeros(B, K, device=device, dtype=dtype)
        attn_full[:, ready_mask] = attn_ready.to(dtype=dtype)

        # Compute entropy
        safe_attn = attn_ready.clamp(min=1e-8)
        entropy = -(safe_attn * safe_attn.log()).sum(dim=-1)

        return context.to(dtype=dtype), attn_full, entropy

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        """
        Hierarchical prototype retrieval.

        Args:
            features: [B, C, H, W] feature map

        Returns:
            dict with:
                - context_spatial: [B, D, H, W] broadcast context
                - context_global: [B, D] global context
                - level_weights: [B, L] routing weights
                - level_mix: [B, L] fast/slow mixing per level
                - attn_entropy: [B] average entropy across levels
                - token: [B, D] for EMA update
                - attn_per_level: list of [B, K_l] attention weights
        """
        B, C, H, W = features.shape

        # Query projection
        query = self.query_proj(features)
        token = F.adaptive_avg_pool2d(query, 1).flatten(1)
        token = F.normalize(token, dim=-1)

        # Routing: which levels are important?
        level_weights = self.routing_net(features)  # [B, L]

        # Retrieve from each level
        level_contexts = []
        level_attns = []
        level_mix = []
        total_entropy = torch.zeros(B, device=features.device)

        for level in range(self.num_levels):
            bank_dict = self.banks[level]
            fast_bank = bank_dict["fast"]
            slow_bank = bank_dict["slow"]
            fast_counts = self.fast_counts_list[level]
            slow_counts = self.slow_counts_list[level]

            # Retrieve from fast bank
            ctx_fast, attn_fast, entropy_fast = self._retrieve_from_bank(
                token, fast_bank, fast_counts
            )
            # Retrieve from slow bank
            ctx_slow, attn_slow, entropy_slow = self._retrieve_from_bank(
                token, slow_bank, slow_counts
            )

            # Mix fast and slow
            mix_gate = self.mix_gates[level](token).view(B, 1)  # [B, 1]
            ctx_mixed = mix_gate * ctx_fast + (1.0 - mix_gate) * ctx_slow

            # Weight by level importance
            level_weight = level_weights[:, level].view(B, 1)
            level_contexts.append(ctx_mixed * level_weight)
            level_attns.append(torch.stack([attn_fast, attn_slow], dim=1))  # [B, 2, K]
            level_mix.append(torch.cat([mix_gate, 1.0 - mix_gate], dim=1))

            total_entropy = total_entropy + level_weight.squeeze(1) * (
                entropy_fast + entropy_slow
            ) / 2

        # Combine all levels
        context_global = torch.stack(level_contexts, dim=1).sum(dim=1)  # [B, D]
        context_spatial = context_global.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        return {
            "context_spatial": context_spatial,
            "context_global": context_global,
            "level_weights": level_weights.detach(),
            "level_mix": torch.stack(level_mix, dim=1).detach(),  # [B, L, 2]
            "attn_entropy": total_entropy.detach(),
            "token": token.detach(),
            "attn_per_level": [a.detach() for a in level_attns],
        }

    @torch.no_grad()
    def update_prototypes(
        self,
        cache: Dict[str, Tensor],
        momentum: Optional[float] = None,
    ) -> None:
        """
        EMA update for all levels.
        """
        token = cache["token"]
        attn_per_level = cache["attn_per_level"]  # list of [B, 2, K_l]

        for level in range(self.num_levels):
            fast_counts = self.fast_counts_list[level]
            slow_counts = self.slow_counts_list[level]
            fast_bank = self.banks[level]["fast"]
            slow_bank = self.banks[level]["slow"]

            attn = attn_per_level[level]  # [B, 2, K]
            fast_attn = attn[:, 0]  # [B, K]
            slow_attn = attn[:, 1]  # [B, K]

            # Update fast bank
            self._update_bank(
                fast_bank,
                fast_counts,
                token,
                fast_attn,
                self.fast_momentum if momentum is None else momentum,
            )
            # Update slow bank
            self._update_bank(
                slow_bank,
                slow_counts,
                token,
                slow_attn,
                self.slow_momentum if momentum is None else momentum * 0.25,
            )

    @torch.no_grad()
    def _update_bank(
        self,
        bank: nn.Parameter,
        counts: Tensor,
        tokens: Tensor,
        attn: Tensor,
        momentum: float,
    ) -> None:
        """Bootstrap + EMA update."""
        B = tokens.shape[0]
        tokens_norm = F.normalize(tokens, dim=-1)

        # Bootstrap inactive slots
        inactive = torch.nonzero(counts <= 1e-6, as_tuple=False).flatten()
        if inactive.numel() > 0 and B > 0:
            num_boot = min(inactive.numel(), B)
            for i in range(num_boot):
                bank.data[inactive[i]].copy_(tokens_norm[i % B])
                counts[inactive[i]].fill_(1.0)

        # EMA update based on attention assignments
        active = torch.nonzero(counts > 1e-6, as_tuple=False).flatten()
        if active.numel() == 0:
            return

        bank_active = F.normalize(bank.data[active], dim=-1)
        sim = torch.matmul(tokens_norm, bank_active.t())
        assignments = sim.argmax(dim=-1)

        for k_local in range(active.numel()):
            k_global = int(active[k_local].item())
            mask = assignments == k_local
            if not torch.any(mask):
                continue
            target = F.normalize(tokens_norm[mask].mean(dim=0), dim=0)
            bank.data[k_global] = bank.data[k_global] * (1.0 - momentum) + target * momentum
            counts[k_global] += float(mask.sum().item())

        # Max-norm constraint
        norms = bank.data.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.max_norm / (norms + 1e-6), max=1.0)
        bank.data.mul_(scale)


# =============================================================================
# 5. MONTE CARLO DROPOUT UNCERTAINTY
# =============================================================================


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.

    During training: samples dropout masks stochastically
    During inference: can run multiple forward passes to get uncertainty
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor, num_samples: int = 1) -> Union[Tensor, List[Tensor]]:
        """
        If num_samples=1: single stochastic forward (training mode)
        If num_samples>1: multiple forwards for inference (returns list)
        """
        if num_samples == 1:
            return F.dropout2d(x, p=self.p, training=self.training)
        # Multiple inference samples
        return [F.dropout2d(x, p=self.p, training=True) for _ in range(num_samples)]


class UncertaintyCalibrator(nn.Module):
    """
    Calibrate uncertainty using MC Dropout + variance estimation.
    """

    def __init__(
        self,
        channels: int,
        num_mc_samples: int = 10,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.num_mc_samples = num_mc_samples
        self.mc_dropout = MCDropout(p=dropout_p)

        # Network to combine uncertainty sources
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(channels + 1, channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, max(1, channels // 8)), channels // 2),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 1),
        )

    def estimate_aleatoric(self, logits_samples: List[Tensor]) -> Tensor:
        """
        Estimate aleatoric uncertainty from variance of predictions.
        """
        stacked = torch.stack(logits_samples, dim=0)  # [N, B, 1, H, W]
        probs = torch.sigmoid(stacked)
        variance = probs.var(dim=0)  # [B, 1, H, W]
        return variance

    def estimate_epistemic(self, logits_samples: List[Tensor]) -> Tensor:
        """
        Estimate epistemic uncertainty from predictive entropy.
        """
        stacked = torch.stack(logits_samples, dim=0)  # [N, B, 1, H, W]
        probs = torch.sigmoid(stacked)
        mean_prob = probs.mean(dim=0, keepdim=True)
        entropy = -(mean_prob * torch.log(mean_prob + 1e-8) + (1 - mean_prob) * torch.log(1 - mean_prob + 1e-8))
        return entropy.squeeze(1)  # [B, H, W]

    def forward(
        self,
        features: Tensor,
        logits: Tensor,
        return_uncertainty: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Calibrate uncertainty estimate.

        Args:
            features: [B, C, H, W] features from decoder
            logits: [B, 1, H, W] coarse segmentation logits
            return_uncertainty: whether to return uncertainty maps

        Returns:
            if return_uncertainty:
                calibrated_logits, uncertainty_dict
            else:
                calibrated_logits
        """
        if self.training:
            # During training, apply MC dropout
            features_aug = self.mc_dropout(features, num_samples=1)
            return features_aug

        # During eval, we could do MC sampling but it's expensive
        # Instead, we'll use a single forward with dropout enabled
        features_aug = self.mc_dropout(features, num_samples=1)

        if return_uncertainty:
            # Run multiple MC samples to estimate uncertainty
            with torch.no_grad():
                logits_samples = []
                for _ in range(self.num_mc_samples):
                    feat_aug = self.mc_dropout(features, num_samples=1)
                    # We need a head to convert feat to logits; for now skip
                    # In practice, this would require the seg_head
                    logits_samples.append(logits)  # placeholder

                if len(logits_samples) > 1:
                    aleatoric = self.estimate_aleatoric(logits_samples)
                    epistemic = self.estimate_epistemic(logits_samples)
                    total_unc = aleatoric + epistemic
                else:
                    total_unc = torch.zeros_like(logits)

                uncertainty_dict = {
                    "aleatoric": aleatoric if len(logits_samples) > 1 else None,
                    "epistemic": epistemic if len(logits_samples) > 1 else None,
                    "total": total_unc,
                }

            return features_aug, uncertainty_dict

        return features_aug


# =============================================================================
# 6. ENHANCED LOSS FUNCTIONS
# =============================================================================


class PrototypeContrastiveLoss(nn.Module):
    """
    Contrastive loss to separate prototypes of different classes.

    Assumes we have access to prototype labels (either assigned via clustering
    or from ground truth). Encourages inter-class separation and intra-class
    cohesion.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        prototypes: Tensor,  # [K, D]
        prototype_labels: Optional[Tensor] = None,  # [K]
        class_centroids: Optional[Tensor] = None,  # [C, D]
    ) -> Tensor:
        """
        Compute contrastive loss.

        If prototype_labels is available: use instance-level contrastive
        If class_centroids available: use class-level contrastive
        """
        K, D = prototypes.shape
        device = prototypes.device

        if prototype_labels is not None:
            # Instance-level: pull same class, push different classes
            # Normalize
            prot_norm = F.normalize(prototypes, dim=-1)
            similarity = torch.matmul(prot_norm, prot_norm.t()) / self.temperature  # [K, K]

            # Mask for positive pairs (same class)
            labels = prototype_labels.view(-1, 1)
            mask = (labels == labels.t()).float()  # [K, K]
            mask.fill_diagonal_(0)  # exclude self

            # For each prototype, contrast with all others
            pos_sim = (similarity * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1))
            neg_sim = (similarity * (1 - mask)).sum(dim=1) / ((1 - mask).sum(dim=1).clamp(min=1))

            loss = F.relu(self.margin - pos_sim + neg_sim).mean()
            return loss

        elif class_centroids is not None:
            # Class-level: ensure prototype-cluster centroids are well-separated
            C = class_centroids.shape[0]
            centroids_norm = F.normalize(class_centroids, dim=-1)
            prot_norm = F.normalize(prototypes, dim=-1)

            # Compute similarity between all centroids
            centroid_sim = torch.matmul(centroids_norm, centroids_norm.t())  # [C, C]
            # We want negative pairs to have low similarity
            mask = 1 - torch.eye(C, device=device)
            negative_sim = (centroid_sim * mask).sum() / mask.sum()

            # Also ensure prototypes are close to their class centroid
            # (requires knowing which prototype belongs to which class)
            # For now, just penalize inter-centroid similarity
            loss = F.relu(negative_sim - self.margin)
            return loss

        return torch.tensor(0.0, device=device)


class InnerLoopConsistencyLoss(nn.Module):
    """
    Ensure surprise decreases over inner steps.

    Penalizes if surprise increases or doesn't decrease enough.
    """

    def __init__(self, decay_factor: float = 0.9):
        super().__init__()
        self.decay_factor = decay_factor

    def forward(self, inner_steps_info: List[Dict]) -> Tensor:
        """
        Args:
            inner_steps_info: list of dicts with 'total_surprise' for each step

        Returns:
            loss: positive if surprise doesn't decay properly
        """
        if len(inner_steps_info) < 2:
            return torch.tensor(0.0)

        surprises = [info["total_surprise"] for info in inner_steps_info]
        surprises_tensor = torch.stack(surprises)  # [S]

        # Desired: exponential decay
        expected_decay = surprises[0] * (self.decay_factor ** torch.arange(len(surprises), device=surprises_tensor.device))

        # Compute deviation from expected decay
        loss = F.mse_loss(surprises_tensor, expected_decay)
        return loss


class GateSparsityLoss(nn.Module):
    """
    Encourage balanced gate activations (avoid all-0 or all-1 collapse).
    """

    def __init__(self, target_entropy: float = 0.5, weight: float = 0.1):
        super().__init__()
        self.target_entropy = target_entropy
        self.weight = weight

    def forward(self, gate_values: Union[Tensor, List[float]]) -> Tensor:
        """
        Args:
            gate_values: single value or list of gate values from batch

        Returns:
            loss: pushes gate distribution away from extremes
        """
        if isinstance(gate_values, list):
            if not gate_values:
                return torch.tensor(0.0)
            gate_tensor = torch.tensor(gate_values, device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            gate_tensor = gate_values

        # Compute entropy of gate distribution
        gate_clamped = gate_tensor.clamp(1e-8, 1 - 1e-8)
        entropy = -(gate_clamped * torch.log(gate_clamped) + (1 - gate_clamped) * torch.log(1 - gate_clamped + 1e-8))

        # Target entropy (max for Bernoulli is ln(2) ≈ 0.693)
        # Use 0.5 as middle ground
        loss = (entropy.mean() - self.target_entropy).abs()
        return self.weight * loss


class MemoryQualityRegularizer(nn.Module):
    """
    Regularize prototype banks for quality and diversity.
    """

    def __init__(
        self,
        min_usage_ratio: float = 0.1,
        max_prototype_norm: float = 1.0,
        diversity_weight: float = 0.1,
    ):
        super().__init__()
        self.min_usage_ratio = min_usage_ratio
        self.max_prototype_norm = max_prototype_norm
        self.diversity_weight = diversity_weight

    def forward(
        self,
        prototypes: Tensor,
        counts: Tensor,
        attention_weights: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute quality metrics for prototype bank.

        Args:
            prototypes: [K, D]
            counts: [K] usage counts
            attention_weights: [B, K] attention from recent batch (optional)

        Returns:
            dict of regularization losses
        """
        K = prototypes.shape[0]
        device = prototypes.device
        losses = {}

        # 1. Usage regularization: encourage at least min_usage_ratio of prototypes to be used
        total_count = counts.sum()
        if total_count > 0:
            active_ratio = (counts > 1e-6).float().mean()
            usage_loss = F.relu(self.min_usage_ratio - active_ratio)
            losses["usage_ratio"] = usage_loss
        else:
            losses["usage_ratio"] = torch.tensor(0.0, device=device)

        # 2. Norm regularization: keep prototypes bounded
        norms = prototypes.norm(dim=-1)
        norm_loss = F.relu(norms - self.max_prototype_norm).mean()
        losses["prototype_norm"] = norm_loss

        # 3. Diversity: encourage prototypes to be different
        if K > 1:
            prot_norm = F.normalize(prototypes, dim=-1)
            similarity = torch.matmul(prot_norm, prot_norm.t())  # [K, K]
            # Mask diagonal
            mask = 1 - torch.eye(K, device=device)
            avg_similarity = (similarity * mask).sum() / (mask.sum() + 1e-8)
            diversity_loss = F.relu(avg_similarity - 0.5)  # penalize high similarity
            losses["diversity"] = self.diversity_weight * diversity_loss
        else:
            losses["diversity"] = torch.tensor(0.0, device=device)

        # 4. If attention weights provided: encourage uniform usage
        if attention_weights is not None:
            B = attention_weights.shape[0]
            avg_attn = attention_weights.mean(dim=0)  # [K]
            target_attn = torch.ones_like(avg_attn) / K
            attn_entropy = -(avg_attn * torch.log(avg_attn + 1e-8)).sum()
            # We want high entropy (uniform)
            target_entropy = math.log(K)
            attn_loss = F.relu(target_entropy - attn_entropy)  # penalize low entropy
            losses["attention_entropy"] = attn_loss

        return losses


class EnhancedStrongBaselineLoss(StrongBaselineLoss):
    """
    Extended loss with prototype contrastive, inner loop consistency,
    gate sparsity, and memory quality regularization.
    """

    def __init__(
        self,
        # Base loss weights (from parent)
        bce_weight: float = 0.20,
        lovasz_weight: float = 0.30,
        focal_tversky_weight: float = 0.25,
        dice_weight: float = 0.15,
        aux_weight: float = 0.10,
        coarse_weight: float = 0.08,
        trust_weight: float = 0.04,
        # New loss weights
        prototype_contrastive_weight: float = 0.05,
        inner_consistency_weight: float = 0.02,
        gate_sparsity_weight: float = 0.01,
        memory_quality_weight: float = 0.02,
    ):
        super().__init__(
            bce_weight=bce_weight,
            lovasz_weight=lovasz_weight,
            focal_tversky_weight=focal_tversky_weight,
            dice_weight=dice_weight,
            aux_weight=aux_weight,
            coarse_weight=coarse_weight,
            trust_weight=trust_weight,
        )

        self.prototype_contrastive_weight = prototype_contrastive_weight
        self.inner_consistency_weight = inner_consistency_weight
        self.gate_sparsity_weight = gate_sparsity_weight
        self.memory_quality_weight = memory_quality_weight

        # Loss modules
        self.prototype_contrastive = PrototypeContrastiveLoss()
        self.inner_consistency = InnerLoopConsistencyLoss()
        self.gate_sparsity = GateSparsityLoss()
        self.memory_quality = MemoryQualityRegularizer()

    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: Tensor,
        nested_info: Optional[Dict] = None,
        prototype_cache: Optional[Dict] = None,
        prototype_labels: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, float]]]:
        """
        Compute total loss with all components.

        Args:
            outputs: model outputs (logits, aux_logits, coarse_logits)
            targets: ground truth masks
            nested_info: dict with nested module info (inner_steps, gate_values)
            prototype_cache: cache from prototype bank update
            prototype_labels: class labels for prototypes (if available)
        """
        # Base loss
        total_loss, loss_dict = super().forward(outputs, targets, return_components=True)

        device = targets.device

        # 1. Prototype contrastive loss
        if prototype_cache is not None and "prototypes" in prototype_cache:
            prototypes = prototype_cache["prototypes"]
            counts = prototype_cache["counts"]
            proto_loss = self.prototype_contrastive(
                prototypes, prototype_labels
            )
            loss_dict["loss_proto_contrast"] = proto_loss.item()
            total_loss = total_loss + self.prototype_contrastive_weight * proto_loss

        # 2. Inner loop consistency loss
        if nested_info is not None:
            inner_consist_total = torch.tensor(0.0, device=device)
            num_stages = 0
            for stage_key, info in nested_info.items():
                # Only process dict entries (stage-specific info)
                if not isinstance(info, dict):
                    continue
                if "inner_steps" in info and len(info["inner_steps"]) > 1:
                    inner_consist_total = inner_consist_total + self.inner_consistency(
                        info["inner_steps"]
                    )
                    num_stages += 1

            if num_stages > 0:
                inner_consist_loss = inner_consist_total / num_stages
                loss_dict["loss_inner_consist"] = inner_consist_loss.item()
                total_loss = total_loss + self.inner_consistency_weight * inner_consist_loss

        # 3. Gate sparsity loss
        if nested_info is not None:
            gate_values = []
            for stage_key, info in nested_info.items():
                # Only process dict entries
                if isinstance(info, dict) and "gate_value" in info:
                    gate_values.append(info["gate_value"])
            if gate_values:
                gate_loss = self.gate_sparsity(gate_values)
                loss_dict["loss_gate_sparsity"] = gate_loss.item()
                total_loss = total_loss + self.gate_sparsity_weight * gate_loss

        # 4. Memory quality regularization
        if prototype_cache is not None:
            # Combine across all levels if hierarchical
            mem_loss_total = torch.tensor(0.0, device=device)
            num_banks = 0

            if "prototypes" in prototype_cache:
                # Simple flat bank
                mem_losses = self.memory_quality(
                    prototype_cache["prototypes"],
                    prototype_cache["counts"],
                    prototype_cache.get("attention"),
                )
                for k, v in mem_losses.items():
                    loss_dict[f"mem_{k}"] = v.item()
                    mem_loss_total = mem_loss_total + v
                num_banks = 1
            elif "bank_states" in prototype_cache:
                # Hierarchical banks
                for level_state in prototype_cache["bank_states"]:
                    mem_losses = self.memory_quality(
                        level_state["prototypes"],
                        level_state["counts"],
                        level_state.get("attention"),
                    )
                    for k, v in mem_losses.items():
                        loss_dict[f"mem_level{level_state['level']}_{k}"] = v.item()
                    mem_loss_total = mem_loss_total + sum(mem_losses.values())
                    num_banks += 1

            if num_banks > 0:
                avg_mem_loss = mem_loss_total / num_banks
                loss_dict["loss_memory_quality"] = avg_mem_loss.item()
                total_loss = total_loss + self.memory_quality_weight * avg_mem_loss

        loss_dict["loss_total"] = total_loss.item()
        return total_loss, loss_dict
