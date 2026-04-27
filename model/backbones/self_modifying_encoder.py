"""
Self-Modifying Encoder — Strategy B: Hope-style Nested Learning
================================================================
Áp dụng Nested Learning paradigm vào encoder cho bài toán polyp segmentation.

Paper: "Nested Learning: The Illusion of Deep Learning Architectures"
       (Behrouz et al., NeurIPS 2025)

Core idea (§5, §7, §8 của paper):
- Mỗi encoder stage là một Neural Learning Module với internal objective riêng
- Tại inference, mỗi stage TỰ CẬP NHẬT weights dựa trên "surprise" của features
- Đây là self-referential process: M_{t+1} = M_t − η · ∇L_inner(M_t; x_t)
- Các stage cập nhật ở tần suất khác nhau (Continuum Memory System)

Integration với PolyMemnet:
- Wrap PVTv2-B2 / ConvNeXt encoder stages
- Mỗi stage output đi qua SelfModifyingBlock trước khi vào FPN decoder
- Kết nối với SafeNestedResidualRefiner qua nested_cache

Author: viethung-pka (PolyMemnet project)
"""

import math
import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# 1. SURPRISE OBJECTIVES — Internal loss cho self-modification
# =============================================================================

class SurpriseObjective(nn.Module): 
    """
    Tính Local Surprise Signal (LSS) cho một feature map.
    
    Từ paper NL §4.1:
        "Backpropagation là associative memory ánh xạ mỗi data point 
         đến error trong prediction tương ứng."
    
    Ở đây ta dùng 2 loại surprise:
    1. Reconstruction surprise: feature map được mask → reconstruct → error
    2. Consistency surprise: feature statistics lệch khỏi running mean → error
    
    Kết hợp cả hai cho robust self-modification signal.
    """

    def __init__(self, channels: int, spatial_mask_ratio: float = 0.25):
        super().__init__()
        self.channels = channels
        self.spatial_mask_ratio = spatial_mask_ratio

        # Lightweight reconstructor (2-layer bottleneck)
        bottleneck = channels // 4
        self.reconstructor = nn.Sequential(
            nn.Conv2d(channels, bottleneck, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(bottleneck, channels, 1, bias=False),
        )

        # Running statistics for consistency surprise
        self.register_buffer("running_mean", torch.zeros(1, channels, 1, 1))
        self.register_buffer("running_var", torch.ones(1, channels, 1, 1))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.ema_momentum = 0.1

    def _spatial_mask(self, x: Tensor) -> Tensor:
        """Tạo random spatial mask cho reconstruction task."""
        B, C, H, W = x.shape
        num_patches = H * W
        num_mask = int(num_patches * self.spatial_mask_ratio)

        # Random mask per sample in batch
        mask = torch.ones(B, 1, H, W, device=x.device)
        for b in range(B):
            indices = torch.randperm(num_patches, device=x.device)[:num_mask]
            rows = indices // W
            cols = indices % W
            mask[b, 0, rows, cols] = 0.0
        return mask  # 1 = keep, 0 = mask

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Tính surprise loss cho feature map x.
        
        Returns:
            loss: scalar surprise loss
            info: dict chứa thông tin debug
        """
        B, C, H, W = x.shape

        # --- Reconstruction surprise ---
        mask = self._spatial_mask(x)
        masked_x = x * mask
        reconstructed = self.reconstructor(masked_x)

        # Loss chỉ tính trên vùng bị mask
        inv_mask = 1.0 - mask
        recon_loss = (((reconstructed - x) ** 2) * inv_mask).sum() / (
            inv_mask.sum() * C + 1e-8
        )

        # --- Consistency surprise ---
        # So sánh feature statistics hiện tại với running statistics
        current_mean = x.mean(dim=[0, 2, 3], keepdim=True)
        current_var = x.var(dim=[0, 2, 3], keepdim=True)

        # Buffers stay fp32; cast current stats to match (autocast compatibility)
        buf_dtype = self.running_mean.dtype
        current_mean_buf = current_mean.detach().to(dtype=buf_dtype)
        current_var_buf = current_var.detach().to(dtype=buf_dtype)

        if self.training:
            with torch.no_grad():
                self.num_batches_tracked += 1
                self.running_mean.lerp_(current_mean_buf, self.ema_momentum)
                self.running_var.lerp_(current_var_buf, self.ema_momentum)

        # KL-divergence-like measure between current and running statistics
        running_mean = self.running_mean.to(dtype=current_mean.dtype)
        running_var = self.running_var.to(dtype=current_var.dtype)
        consist_loss = (
            (current_mean - running_mean).pow(2).mean()
            + (current_var / (running_var + 1e-8) - 1).pow(2).mean()
        )

        # Combined surprise
        total_surprise = recon_loss + 0.1 * consist_loss

        info = {
            "recon_loss": recon_loss.detach(),
            "consist_loss": consist_loss.detach(),
            "total_surprise": total_surprise.detach(),
        }
        return total_surprise, info


# =============================================================================
# 2. SELF-MODIFYING BLOCK — Core NL component
# =============================================================================

class SelfModifyingBlock(nn.Module):
    """
    Self-Modifying Block theo Hope architecture (NL paper §8).
    
    Mỗi block có:
    - Một bộ weights chính (persistent, Level 1)
    - Một bộ weights modifier (adaptive, Level 2) — tự cập nhật qua inner loop
    - Internal objective (surprise-based) để drive self-modification
    
    Tại forward pass:
    1. Tính feature output từ persistent weights
    2. Tính surprise loss trên features  
    3. Chạy K inner gradient steps trên modifier weights
    4. Apply modified features qua residual connection
    
    Mapping vào NL formulation:
        Level 1 (outer): W_persistent — trained bởi outer optimizer (AdamW)
        Level 2 (inner): W_modifier — self-updated mỗi forward pass
        
        W_modifier_{t+1} = W_modifier_t − η_inner · ∇L_surprise(W_modifier_t; features)
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
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr

        # === Level 1: Persistent weights (trained by outer optimizer) ===
        # Gating: quyết định mức độ modification được apply
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, 1),
            nn.Sigmoid(),
        )

        # === Level 2: Self-modifying weights (updated per forward pass) ===
        hidden = channels * modifier_expansion
        self.modifier = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(min(8, hidden // 4), hidden),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        # Surprise objective cho inner loop
        self.surprise = SurpriseObjective(channels)

        # Meta-learned initial state cho modifier (NL §3.3 — Knowledge Transfer via Init)
        # Lưu bản sao ban đầu để reset modifier mỗi forward pass
        self._modifier_init_state = None

        # Scale factor cho residual (bắt đầu nhỏ, tăng dần)
        self.residual_scale = nn.Parameter(torch.tensor(0.05))

        # Inner learning rate (adjusted per-stage, not trained by outer optimizer)
        self.register_buffer("log_inner_lr", torch.tensor(math.log(inner_lr)))

    def _get_inner_lr(self) -> float:
        return self.log_inner_lr.exp().item()

    def _save_modifier_init(self):
        """Lưu meta-learned initial state."""
        self._modifier_init_state = {
            name: param.data.clone()
            for name, param in self.modifier.named_parameters()
        }

    def _reset_modifier(self):
        """Reset modifier về meta-learned initial state."""
        if self._modifier_init_state is not None:
            for name, param in self.modifier.named_parameters():
                param.data.copy_(self._modifier_init_state[name])

    def _inner_loop(self, features: Tensor) -> Tuple[Tensor, Tensor, Dict]:
        """
        Self-referential inner optimization loop (Hope-style / TTT-style).
        
        Paper NL Eq. formulation:
            M_{t+1} = M_t − η · ∇L_inner(M_t; x_t)
            
        Ở đây:
            M = self.modifier (bộ weights adaptive)
            L_inner = surprise objective  
            x_t = features từ encoder stage
            
        Implementation:
            - Inner loop LUÔN chạy với gradients enabled (kể cả tại inference)
              vì self-modification là bản chất của module, không phải training artifact.
            - Outer gradient flow qua final forward pass bình thường.
            - Surprise loss được trả về để train reconstructor qua outer loop.
            
        Tại inference (torch.no_grad() bên ngoài):
            - torch.enable_grad() bên trong cho phép inner optimization vẫn hoạt động
            - modifier tự adapt per-image
            - Nhưng outer graph KHÔNG được tạo (tiết kiệm memory)
        """
        inner_lr = self._get_inner_lr()
        all_info = []

        # Inner loop CẦN gradients để self-modify, kể cả khi eval
        with torch.enable_grad():
            # Đảm bảo features có gradient tracking cho inner loop
            feat_inner = features.detach().requires_grad_(False)

            for step in range(self.inner_steps):
                modified = self.modifier(feat_inner)
                surprise_loss, info = self.surprise(modified + feat_inner)
                info["inner_step"] = step
                all_info.append(info)

                if step < self.inner_steps - 1:
                    # Inner gradient step trên modifier weights
                    mod_params = [p for p in self.modifier.parameters() if p.requires_grad]
                    if mod_params and surprise_loss.requires_grad:
                        grads = torch.autograd.grad(
                            surprise_loss,
                            mod_params,
                            create_graph=False,
                            retain_graph=False,
                            allow_unused=True,
                        )
                        with torch.no_grad():
                            for param, grad in zip(mod_params, grads):
                                if grad is not None:
                                    param.data.sub_(inner_lr * grad)

        # Final forward pass dùng features GỐC (giữ outer graph nếu training)
        final_modified = self.modifier(features)

        # Auxiliary surprise loss cho outer loop (trains reconstructor)
        # Luôn compute vì graph chạy qua modifier params → reconstructor params
        if any(p.requires_grad for p in self.modifier.parameters()):
            aux_surprise, _ = self.surprise(final_modified + features)
        else:
            aux_surprise = torch.tensor(0.0, device=features.device)

        return final_modified, aux_surprise, all_info

    def forward(
        self, x: Tensor, return_info: bool = False
    ) -> Tuple[Tensor, Optional[Dict]]:
        """
        Forward pass với self-modification.
        
        Flow:
        1. Save modifier state (sẽ được modified in-place)
        2. Run inner loop: modifier tự cập nhật K steps
        3. Compute gated residual output
        4. Reset modifier về initial state (cho sample tiếp theo)
        
        Args:
            x: feature map [B, C, H, W] từ encoder stage
            return_info: có trả về debug info không
            
        Returns:
            output: modified feature map [B, C, H, W]
            info: dict với surprise metrics (nếu return_info=True)
        """
        # Bước 1: Save state trước khi inner loop modify
        saved_state = {
            name: param.data.clone()
            for name, param in self.modifier.named_parameters()
        }

        # Bước 2: Inner loop — modifier tự cập nhật
        modification, aux_surprise, all_info = self._inner_loop(x)

        # Bước 3: Gated residual connection
        # Gate quyết định bao nhiêu modification được apply
        gate_value = self.gate(x)  # [B, 1]
        gate_value = gate_value.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]

        # Residual: x + scale * gate * modification
        output = x + self.residual_scale * gate_value * modification

        # Bước 4: Reset modifier weights cho sample tiếp theo
        for name, param in self.modifier.named_parameters():
            param.data.copy_(saved_state[name])

        info = None
        if return_info:
            info = {
                "gate_value": gate_value.detach().mean().item(),
                "residual_scale": self.residual_scale.item(),
                "inner_lr": self._get_inner_lr(),
                "inner_steps": all_info,
                "aux_surprise": aux_surprise,  # For outer loss
            }

        return output, info


# =============================================================================
# 3. SELF-MODIFYING ENCODER — Wrap toàn bộ encoder
# =============================================================================

class SelfModifyingEncoder(nn.Module):
    """
    Wrap một pre-trained encoder (PVTv2-B2, ConvNeXt, Swin) với
    Self-Modifying Blocks ở mỗi stage output.
    
    NL Paradigm mapping:
    ┌─────────────────────────────────────────────────────────────┐
    │ Level 0 (outer-most): Pre-training objective trên ImageNet  │
    │   → Persistent encoder weights (frozen hoặc slow fine-tune) │
    │                                                             │
    │ Level 1 (outer): Segmentation loss (BCE + Dice + Lovász)    │
    │   → Persistent projections, gates, residual scales          │
    │                                                             │
    │ Level 2 (inner): Surprise objectives (per-stage)            │
    │   → Self-modifying weights, updated every forward pass      │
    │                                                             │
    │ Frequency: f_level0 = 0 (frozen)                            │
    │            f_level1 = 1/epoch (outer training)              │
    │            f_level2 = 1/sample (inner loop per image)       │
    └─────────────────────────────────────────────────────────────┘
    
    Continuum Memory System (CMS) design:
    - c2 (1/4):  inner_steps=1, fast spatial features (ít cần adapt)
    - c3 (1/8):  inner_steps=2
    - c4 (1/16): inner_steps=3, moderate semantic
    - c5 (1/32): inner_steps=4, cần adapt nhiều nhất (high-level semantic)
    
    Args:
        backbone: pre-trained encoder (PVTv2, ConvNeXt, etc.)
        feature_channels: list of channels cho c2, c3, c4, c5
        inner_steps_schedule: list of inner steps cho mỗi stage
        inner_lr: base inner learning rate
        apply_stages: list of stage indices to apply self-modification
                      (mặc định [2, 3] = chỉ c4, c5 để tiết kiệm compute)
        freeze_backbone: có freeze backbone weights không
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_channels: List[int] = [64, 128, 320, 512],
        inner_steps_schedule: List[int] = [1, 2, 3, 4],
        inner_lr: float = 0.001,
        apply_stages: Optional[List[int]] = None,
        freeze_backbone: bool = False,
        modifier_expansion: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.apply_stages = apply_stages if apply_stages is not None else [2, 3]

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Tạo self-modifying blocks cho các stages được chọn
        self.modifiers = nn.ModuleDict()
        for stage_idx in self.apply_stages:
            ch = feature_channels[stage_idx]
            steps = inner_steps_schedule[stage_idx]
            # Inner LR giảm dần cho stages sâu hơn (slow learner)
            stage_lr = inner_lr * (0.5 ** stage_idx)

            self.modifiers[f"stage_{stage_idx}"] = SelfModifyingBlock(
                channels=ch,
                inner_steps=steps,
                inner_lr=stage_lr,
                modifier_expansion=modifier_expansion,
                dropout=dropout,
            )

        # Layer norm trước khi đưa vào modifier (stabilize)
        self.pre_norms = nn.ModuleDict()
        for stage_idx in self.apply_stages:
            ch = feature_channels[stage_idx]
            self.pre_norms[f"stage_{stage_idx}"] = nn.GroupNorm(
                min(8, ch // 4), ch
            )

    def forward(
        self, x: Tensor, return_nested_info: bool = False
    ) -> Tuple[List[Tensor], Dict]:
        """
        Forward pass:
        1. Backbone extracts c2, c3, c4, c5
        2. Selected stages đi qua SelfModifyingBlock
        3. Return modified features + nested_info cho refiner
        
        Args:
            x: input image [B, 3, H, W]
            return_nested_info: có trả nested info để truyền cho refiner
            
        Returns:
            features: list of [c2, c3, c4, c5] feature maps
            nested_info: dict chứa surprise metrics, gate values, etc.
        """
        # Bước 1: Backbone forward
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        # features = [c2, c3, c4, c5] from backbone
        if isinstance(features, (tuple, list)):
            features = list(features)
        else:
            raise ValueError("Backbone must return list/tuple of feature maps")

        nested_info = {}

        # Bước 2: Apply self-modification trên selected stages
        for stage_idx in self.apply_stages:
            key = f"stage_{stage_idx}"
            if key in self.modifiers:
                # Pre-normalize
                feat = self.pre_norms[key](features[stage_idx])

                # Self-modifying forward (inner loop happens here)
                modified_feat, info = self.modifiers[key](
                    feat, return_info=return_nested_info
                )

                # Replace feature with modified version
                features[stage_idx] = modified_feat

                if return_nested_info and info is not None:
                    nested_info[key] = info

        return features, nested_info


# =============================================================================
# 4. FULL MODEL INTEGRATION — PolyMemnet + Self-Modifying Encoder
# =============================================================================

class PolyMemnetWithNL(nn.Module):
    """
    PolyMemnet với Self-Modifying Encoder (Strategy B — Nested Learning).
    
    Thay thế encoder block trong kiến trúc gốc:
    
    TRƯỚC: Input → Encoder(static) → c2-c5 → BiFPN → seg_head → Refiner → Output
    SAU:   Input → SelfModifyingEncoder → c2-c5(adapted) → BiFPN → seg_head → Refiner → Output
                        ↑                                                        ↑
                   Level 2 inner loop                              Level 2 memory bank
                   (surprise-based)                                (prototype-based)
                        └─────────── nested_info ─────────────────────┘
    
    Hai hệ thống Level 2 (encoder self-mod + refiner memory) kết nối qua
    nested_info, tạo thành multi-level nested system hoàn chỉnh.
    """

    def __init__(
        self,
        backbone: nn.Module,
        decoder: nn.Module,
        seg_head: nn.Module,
        refiner: nn.Module,
        aux_head: Optional[nn.Module] = None,
        feature_channels: List[int] = [64, 128, 320, 512],
        inner_steps_schedule: List[int] = [1, 2, 3, 4],
        inner_lr: float = 0.01,
        apply_stages: Optional[List[int]] = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Wrap encoder với self-modification
        self.encoder = SelfModifyingEncoder(
            backbone=backbone,
            feature_channels=feature_channels,
            inner_steps_schedule=inner_steps_schedule,
            inner_lr=inner_lr,
            apply_stages=apply_stages,
            freeze_backbone=freeze_backbone,
        )

        self.decoder = decoder
        self.seg_head = seg_head
        self.refiner = refiner
        self.aux_head = aux_head

    def forward(
        self, x: Tensor, gt_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Full forward pass.
        
        Args:
            x: input image [B, 3, H, W]
            gt_mask: ground truth mask (for training refiner memory)
            
        Returns:
            dict with: coarse_logits, logits, aux_logits, nested_info, nested_cache
        """
        input_size = x.shape[2:]

        # 1. Self-Modifying Encoder — inner loop happens here
        features, nested_info = self.encoder(x, return_nested_info=True)

        # 2. FPN Decoder
        fused_feat, aux_feat = self.decoder(features)

        # 3. Segmentation Head → coarse logits
        coarse_logits = self.seg_head(F.dropout2d(fused_feat, p=0.1, training=self.training))

        # 4. Nested Residual Refiner
        # Truyền nested_info từ encoder vào refiner để tạo inter-level connection
        if hasattr(self.refiner, "forward_with_nested_info"):
            refined_logits, refiner_cache = self.refiner.forward_with_nested_info(
                coarse_logits, fused_feat, nested_info=nested_info, gt_mask=gt_mask
            )
        else:
            # Fallback cho refiner cũ không hỗ trợ nested_info
            refined_logits = self.refiner(coarse_logits, fused_feat, gt_mask=gt_mask)
            refiner_cache = {}

        # 5. Auxiliary Head
        aux_logits = None
        if self.aux_head is not None and aux_feat is not None:
            aux_logits = self.aux_head(aux_feat)

        # 6. Upsample tất cả về input size
        coarse_logits = F.interpolate(
            coarse_logits, size=input_size, mode="bilinear", align_corners=False
        )
        refined_logits = F.interpolate(
            refined_logits, size=input_size, mode="bilinear", align_corners=False
        )
        if aux_logits is not None:
            aux_logits = F.interpolate(
                aux_logits, size=input_size, mode="bilinear", align_corners=False
            )

        return {
            "coarse_logits": coarse_logits,
            "logits": refined_logits,
            "aux_logits": aux_logits,
            "nested_info": nested_info,
            "nested_cache": refiner_cache,
        }


# =============================================================================
# 5. NESTED LEARNING LOSS — Outer + Inner supervision
# =============================================================================

class NestedLearningLoss(nn.Module):
    """
    Loss function tích hợp NL paradigm.
    
    Ngoài 7-component loss hiện tại của PolyMemnet, thêm:
    - Surprise regularization: đảm bảo inner loop đang học meaningful modifications
    - Gate diversity loss: ngăn gate collapse (tất cả gate = 0 hoặc = 1)
    - Inner consistency loss: features sau modification phải preserve semantic
    
    Mapping vào NL formulation:
    - L_outer = segmentation losses (BCE + Dice + Lovász + ...)
    - L_inner = surprise losses (reconstruction + consistency)
    - L_nested = L_outer + λ · L_inner_regularization
    """

    def __init__(
        self,
        outer_loss_fn: nn.Module,
        surprise_weight: float = 0.05,
        gate_diversity_weight: float = 0.01,
    ):
        super().__init__()
        self.outer_loss_fn = outer_loss_fn
        self.surprise_weight = surprise_weight
        self.gate_diversity_weight = gate_diversity_weight

    def forward(
        self,
        predictions: Dict[str, Tensor],
        gt_mask: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Tính tổng loss.
        
        Args:
            predictions: output dict từ PolyMemnetWithNL
            gt_mask: ground truth segmentation mask
        """
        # Outer loss (7-component segmentation loss hiện tại)
        outer_loss = self.outer_loss_fn(predictions, gt_mask)

        loss_dict = {"outer_loss": outer_loss.detach()}

        # Inner loss regularization từ nested_info
        nested_info = predictions.get("nested_info", {})
        inner_reg = torch.tensor(0.0, device=gt_mask.device)
        gate_div = torch.tensor(0.0, device=gt_mask.device)
        aux_surprise_total = torch.tensor(0.0, device=gt_mask.device)

        for stage_key, info in nested_info.items():
            # Surprise should decrease qua inner steps
            if "inner_steps" in info and len(info["inner_steps"]) > 1:
                first_surprise = info["inner_steps"][0]["total_surprise"]
                last_surprise = info["inner_steps"][-1]["total_surprise"]
                inner_reg = inner_reg + F.relu(last_surprise - first_surprise)

            # Gate diversity: penalize nếu gate quá gần 0 hoặc 1
            if "gate_value" in info:
                g = info["gate_value"]
                gate_div = gate_div + -g * math.log(g + 1e-8) - (1 - g) * math.log(
                    1 - g + 1e-8
                )

            # Auxiliary surprise loss (trains reconstructor via outer optimizer)
            if "aux_surprise" in info:
                aux_surprise_total = aux_surprise_total + info["aux_surprise"]

        num_stages = max(len(nested_info), 1)
        inner_reg = inner_reg / num_stages
        gate_div = gate_div / num_stages
        aux_surprise_total = aux_surprise_total / num_stages

        total_loss = (
            outer_loss
            + self.surprise_weight * (inner_reg + aux_surprise_total)
            + self.gate_diversity_weight * gate_div
        )

        loss_dict["inner_reg"] = inner_reg.detach()
        loss_dict["aux_surprise"] = aux_surprise_total.detach()
        loss_dict["gate_diversity"] = gate_div.detach()
        loss_dict["total_loss"] = total_loss.detach()

        return total_loss, loss_dict


# =============================================================================
# 6. FACTORY & UTILITY FUNCTIONS
# =============================================================================

def build_self_modifying_encoder(
    backbone: nn.Module,
    backbone_name: str = "pvtv2_b2",
    apply_stages: Optional[List[int]] = None,
    inner_steps_schedule: Optional[List[int]] = None,
    inner_lr: float = 0.01,
    freeze_backbone: bool = False,
) -> SelfModifyingEncoder:
    """
    Factory function để tạo SelfModifyingEncoder.
    
    Args:
        backbone: pre-trained backbone module
        backbone_name: tên backbone để xác định feature channels
        apply_stages: stages nào được self-modify (default: [2, 3] = c4, c5)
        inner_steps_schedule: số inner steps cho mỗi stage
        inner_lr: base learning rate cho inner loop
        freeze_backbone: có freeze backbone không
    """
    # Feature channels cho các backbone phổ biến
    CHANNEL_CONFIGS = {
        "pvtv2_b2": [64, 128, 320, 512],
        "pvtv2_b3": [64, 128, 320, 512],
        "pvtv2_b5": [64, 128, 320, 512],
        "convnext_tiny": [96, 192, 384, 768],
        "convnext_small": [96, 192, 384, 768],
        "convnext_base": [128, 256, 512, 1024],
        "swin_tiny": [96, 192, 384, 768],
        "swin_small": [96, 192, 384, 768],
        "swin_base": [128, 256, 512, 1024],
    }

    channels = CHANNEL_CONFIGS.get(backbone_name, [64, 128, 320, 512])

    if apply_stages is None:
        apply_stages = [2, 3]  # Mặc định chỉ c4, c5

    if inner_steps_schedule is None:
        # CMS: tăng dần inner steps cho stages sâu hơn
        inner_steps_schedule = [1, 2, 3, 4]

    return SelfModifyingEncoder(
        backbone=backbone,
        feature_channels=channels,
        inner_steps_schedule=inner_steps_schedule,
        inner_lr=inner_lr,
        apply_stages=apply_stages,
        freeze_backbone=freeze_backbone,
    )


def get_parameter_groups(
    model: PolyMemnetWithNL,
    backbone_lr: float = 1e-4,
    decoder_lr: float = 3e-4,
    modifier_lr: float = 5e-4,
    weight_decay: float = 1e-4,
) -> List[Dict]:
    """
    Tạo parameter groups với learning rates khác nhau.
    
    NL insight: mỗi level cần learning rate phù hợp.
    - Backbone (Level 0): thấp nhất (pre-trained, ít thay đổi)
    - Decoder + Heads (Level 1): trung bình
    - Modifier projections + gates (Level 1): cao hơn (cần train nhanh)
    
    Lưu ý: modifier.modifier weights (Level 2) KHÔNG nằm trong outer optimizer
    vì chúng được tự cập nhật bởi inner loop.
    """
    backbone_params = []
    decoder_params = []
    modifier_outer_params = []  # Level 1 parts of modifier (gates, projections)
    modifier_inner_params = []  # Level 2 parts (self.modifier in SelfModifyingBlock)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "encoder.backbone" in name:
            backbone_params.append(param)
        elif "encoder.modifiers" in name:
            if ".modifier." in name:
                # Đây là inner loop parameters — vẫn cần outer gradient
                # vì create_graph=True trong inner loop
                modifier_inner_params.append(param)
            else:
                modifier_outer_params.append(param)
        elif "encoder.pre_norms" in name:
            modifier_outer_params.append(param)
        else:
            decoder_params.append(param)

    return [
        {
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
            "name": "backbone",
        },
        {
            "params": decoder_params,
            "lr": decoder_lr,
            "weight_decay": weight_decay,
            "name": "decoder_heads",
        },
        {
            "params": modifier_outer_params,
            "lr": modifier_lr,
            "weight_decay": weight_decay,
            "name": "modifier_outer",
        },
        {
            "params": modifier_inner_params,
            "lr": modifier_lr * 0.5,
            "weight_decay": weight_decay * 0.1,  # Less decay for inner params
            "name": "modifier_inner",
        },
    ]


# =============================================================================
# 7. DEMO & TEST
# =============================================================================

def _test_self_modifying_block():
    """Quick smoke test."""
    print("=" * 60)
    print("Testing SelfModifyingBlock...")
    print("=" * 60)

    block = SelfModifyingBlock(channels=320, inner_steps=3, inner_lr=0.01)
    x = torch.randn(2, 320, 24, 24)

    # Training mode
    block.train()
    out, info = block(x, return_info=True)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Gate:   {info['gate_value']:.4f}")
    print(f"  Scale:  {info['residual_scale']:.4f}")
    print(f"  Inner LR: {info['inner_lr']:.6f}")
    for step_info in info["inner_steps"]:
        print(
            f"  Step {step_info['inner_step']}: "
            f"recon={step_info['recon_loss']:.4f}, "
            f"consist={step_info['consist_loss']:.4f}"
        )

    # Verify gradients flow through (output + aux_surprise)
    loss = out.mean() + 0.05 * info["aux_surprise"]
    loss.backward()
    
    grad_status = {}
    for name, p in block.named_parameters():
        if p.requires_grad:
            grad_status[name.split(".")[0]] = p.grad is not None
    
    no_grad = [k for k, v in grad_status.items() if not v]
    print(f"  Gradient flow: {'ALL OK' if not no_grad else f'MISSING: {no_grad}'}")
    print()


def _test_self_modifying_encoder():
    """Test with dummy backbone."""
    print("=" * 60)
    print("Testing SelfModifyingEncoder...")
    print("=" * 60)

    # Dummy backbone that returns 4 feature maps
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)

        def forward(self, x):
            B = x.shape[0]
            return [
                torch.randn(B, 64, 96, 96),    # c2: 1/4
                torch.randn(B, 128, 48, 48),   # c3: 1/8
                torch.randn(B, 320, 24, 24),   # c4: 1/16
                torch.randn(B, 512, 12, 12),   # c5: 1/32
            ]

    backbone = DummyBackbone()
    encoder = SelfModifyingEncoder(
        backbone=backbone,
        feature_channels=[64, 128, 320, 512],
        inner_steps_schedule=[1, 2, 3, 4],
        inner_lr=0.01,
        apply_stages=[2, 3],  # Chỉ modify c4, c5
    )

    x = torch.randn(2, 3, 384, 384)
    encoder.train()
    features, nested_info = encoder(x, return_nested_info=True)

    print(f"  Input: {x.shape}")
    for i, f in enumerate(features):
        print(f"  c{i+2}: {f.shape}")
    print(f"  Nested info keys: {list(nested_info.keys())}")
    for key, info in nested_info.items():
        print(f"    {key}: gate={info['gate_value']:.4f}, inner_steps={len(info['inner_steps'])}")

    # Verify gradients
    total = sum(f.mean() for f in features)
    total.backward()
    print(f"  Total params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Trainable params: {sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}")
    print()


if __name__ == "__main__":
    _test_self_modifying_block()
    _test_self_modifying_encoder()
    print("All tests passed!")