"""
CMS-aware SelfModifyingEncoder — Continuum Memory System cho encoder
======================================================================

Thay thế `SelfModifyingEncoder` cũ (apply_stages flat + inner_lr * 0.5^stage).
Tuân thủ đúng CMS principle từ paper Behrouz et al. (NeurIPS 2025):

    Frequency:   f_stem  <  f_c2  <  f_c3  <  f_c4  <  f_c5
    (càng shallow càng "slow/outer", càng deep càng "fast/inner")

Các thay đổi chính so với bản cũ:

1. PER-STAGE CONFIG rõ ràng
   - inner_steps_schedule      : [0, 0, 1, 2, 4]   # stem, c2, c3, c4, c5
   - inner_lr_schedule         : [0, 0, 1e-3, 5e-3, 2e-2]
   - inner_momentum_schedule   : [0, 0, 0.0, 0.9, 0.95]
   - modifier_expansion_schedule: [0, 0, 1, 2, 4]
   - surprise_type_schedule    : ['none', 'none', 'consistency', 'full', 'full']
   Inner_steps=0 nghĩa là stage đó CHỈ được outer-train (không có inner loop).

2. LAYER-WISE LR DECAY cho backbone (outer side của CMS)
   - Trả về parameter groups với LR giảm dần theo depth
   - backbone.stem         : lr_base * decay^4  (≈ 1.6e-6 nếu base=1e-4, decay=0.5)
   - backbone.stage0 (→c2) : lr_base * decay^3
   - backbone.stage1 (→c3) : lr_base * decay^2
   - backbone.stage2 (→c4) : lr_base * decay^1
   - backbone.stage3 (→c5) : lr_base * decay^0 = lr_base

3. PROGRESSIVE UNFREEZE
   - set_epoch(epoch) điều khiển requires_grad theo schedule:
     * epoch < unfreeze_deep_epoch:   FREEZE toàn bộ backbone, chỉ train adaptor/modifier
     * epoch < unfreeze_mid_epoch:    UNFREEZE stage2, stage3 (c4, c5)
     * epoch < unfreeze_shallow_epoch:UNFREEZE stage1 thêm (c3)
     * epoch ≥ unfreeze_shallow_epoch:UNFREEZE toàn bộ với LLRD

4. INNER OPTIMIZER với PERSISTENT MOMENTUM cho deep stages
   - Plain SGD (bản cũ) thay bằng SGD-with-momentum
   - Momentum buffer được giữ qua các batch cho stage sâu → "running memory"
   - Stage nông thì reset momentum mỗi sample (nếu có inner loop)

5. SURPRISE OBJECTIVE PHÂN CẤP
   - 'none'        : không inner loop, stage chỉ outer-train
   - 'consistency' : chỉ penalize BN/GroupNorm running-stat drift (nhẹ, cheap)
   - 'full'        : masked reconstruction + consistency (cũ)

Tương thích ngược:
- Interface forward(x, return_nested_info=True) giữ nguyên
- nested_info dict có cùng keys như bản cũ để NLLossWrapper không vỡ

Author: viethung-pka (PolyMemnet v2.5, CMS-aware encoder)
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# 1. SURPRISE OBJECTIVES — multi-level
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
# 2. SELF-MODIFYING BLOCK với persistent momentum
# =============================================================================


class CMSSelfModifyingBlock(nn.Module):
    """Self-Modifying Block với inner loop SGD+momentum persistent.

    Điểm khác biệt so với bản cũ:
    - Inner optimizer là SGD + momentum (không còn plain SGD)
    - `persist_momentum=True` → momentum buffer giữ giữa các batch
      (dùng cho stage sâu = "slow-fast memory" của CMS)
    - `persist_momentum=False` → reset mỗi sample (stage nông)
    - Surprise objective có thể là 'consistency' (rẻ) hoặc 'full' (đắt)
    """

    def __init__(
        self,
        channels: int,
        inner_steps: int = 3,
        inner_lr: float = 0.01,
        inner_momentum: float = 0.9,
        modifier_expansion: int = 2,
        dropout: float = 0.1,
        surprise_type: str = "full",
        persist_momentum: bool = True,
        residual_init: float = 0.05,
    ):
        super().__init__()
        assert inner_steps >= 1, "inner_steps phải ≥ 1. Dùng identity block nếu =0."
        assert surprise_type in ("consistency", "full"), f"Unknown surprise: {surprise_type}"

        self.channels = channels
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.inner_momentum = inner_momentum
        self.persist_momentum = persist_momentum
        self.surprise_type = surprise_type

        # Level 1 — outer-trained gate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // 4, 16)),
            nn.GELU(),
            nn.Linear(max(channels // 4, 16), 1),
            nn.Sigmoid(),
        )

        # Level 2 — self-modifying weights (inner loop target)
        hidden = channels * modifier_expansion
        self.modifier = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(min(8, max(1, hidden // 4)), hidden),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        # Surprise objective
        if surprise_type == "consistency":
            self.surprise = ConsistencySurprise(channels)
        else:
            self.surprise = FullSurprise(channels)

        # Residual scale (Level 1, learnable)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

        # Persistent momentum buffer cho inner optimizer (nếu bật)
        # Được init lazy để khớp với dtype/device của modifier params.
        self._momentum_buffer: Optional[Dict[str, Tensor]] = None

    # ------------------------------------------------------------------
    # Momentum buffer helpers
    # ------------------------------------------------------------------
    def _ensure_momentum_buffer(self):
        if not self.persist_momentum:
            return
        if self._momentum_buffer is None:
            self._momentum_buffer = {
                name: torch.zeros_like(p.data)
                for name, p in self.modifier.named_parameters()
            }

    def reset_momentum(self):
        """Gọi giữa các epoch nếu muốn — thường không cần."""
        if self._momentum_buffer is not None:
            for buf in self._momentum_buffer.values():
                buf.zero_()

    # ------------------------------------------------------------------
    # Inner loop với SGD + momentum
    # ------------------------------------------------------------------
    def _inner_loop(self, features: Tensor) -> Tuple[Tensor, Tensor, List[Dict]]:
        all_info: List[Dict] = []
        self._ensure_momentum_buffer()

        with torch.enable_grad():
            feat_inner = features.detach()
            for step in range(self.inner_steps):
                modified = self.modifier(feat_inner)
                surprise_loss, info = self.surprise(modified + feat_inner)
                info["inner_step"] = step
                all_info.append(info)

                if step < self.inner_steps - 1:
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
                                param.data.sub_(self.inner_lr * buf)
                            else:
                                param.data.sub_(self.inner_lr * grad)

        # Final forward dùng features gốc để giữ outer graph
        final_modified = self.modifier(features)
        if any(p.requires_grad for p in self.modifier.parameters()):
            aux_surprise, _ = self.surprise(final_modified + features)
        else:
            aux_surprise = torch.tensor(0.0, device=features.device, dtype=features.dtype)

        return final_modified, aux_surprise, all_info

    # ------------------------------------------------------------------
    # Forward: save state → inner loop → residual → restore
    # ------------------------------------------------------------------
    def forward(self, x: Tensor, return_info: bool = False):
        # Snapshot modifier weights (inner loop sẽ mutate in-place)
        saved_state = {
            name: p.data.clone() for name, p in self.modifier.named_parameters()
        }

        modification, aux_surprise, steps_info = self._inner_loop(x)

        gate = self.gate(x).unsqueeze(-1).unsqueeze(-1)
        output = x + self.residual_scale * gate * modification

        # Restore để sample tiếp theo có cùng meta-learned init
        # (ngoại trừ momentum buffer, nếu persist thì KHÔNG reset)
        for name, p in self.modifier.named_parameters():
            p.data.copy_(saved_state[name])

        info = None
        if return_info:
            info = {
                "gate_value": float(gate.detach().mean().item()),
                "residual_scale": float(self.residual_scale.item()),
                "inner_lr": self.inner_lr,
                "inner_steps": steps_info,
                "aux_surprise": aux_surprise,
                "surprise_type": self.surprise_type,
                "persist_momentum": self.persist_momentum,
            }
        return output, info


# =============================================================================
# 3. LIGHT ADAPTOR cho stage nông (thay cho modifier full)
# =============================================================================


class LightAdaptor(nn.Module):
    """Adaptor dạng LoRA-like cho stage nông: rất ít param, không có inner loop.

    Được train bởi outer optimizer như Level 1 thuần. Dùng khi ta muốn
    c2, c3 thích ứng chút ít với polyp distribution mà không phá prior
    ImageNet quá nhiều.
    """

    def __init__(self, channels: int, rank: int = 4, residual_init: float = 0.05):
        super().__init__()
        self.down = nn.Conv2d(channels, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, channels, 1, bias=False)
        nn.init.kaiming_normal_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(self, x: Tensor, return_info: bool = False):
        delta = self.up(F.gelu(self.down(x)))
        out = x + self.residual_scale * delta
        if return_info:
            info = {
                "gate_value": 1.0,  # không có gate, luôn on
                "residual_scale": float(self.residual_scale.item()),
                "inner_lr": 0.0,
                "inner_steps": [],
                "aux_surprise": torch.zeros((), device=x.device, dtype=x.dtype),
                "surprise_type": "none",
                "persist_momentum": False,
            }
            return out, info
        return out, None


# =============================================================================
# 4. CMS SELF-MODIFYING ENCODER
# =============================================================================


# Default per-stage config theo nguyên lý CMS.
# stage_idx 0..3 tương ứng c2 (stride 4), c3 (stride 8), c4 (stride 16), c5 (stride 32).
# Dùng 'none' cho stage shallow nghĩa là KHÔNG gắn adaptor gì cả
# (chỉ backbone outer-trained). 'light' = LoRA. 'full' = SelfModifyingBlock.
DEFAULT_STAGE_CONFIG: List[Dict[str, Any]] = [
    # c2 — shallow, stable
    {
        "mode": "none",
    },
    # c3 — mid-low, light adaptor
    {
        "mode": "light",
        "lora_rank": 4,
        "residual_init": 0.05,
    },
    # c4 — mid-deep, moderate inner loop
    {
        "mode": "full",
        "inner_steps": 2,
        "inner_lr": 5e-3,
        "inner_momentum": 0.9,
        "modifier_expansion": 2,
        "surprise_type": "full",
        "persist_momentum": True,
        "residual_init": 0.05,
    },
    # c5 — deepest, strongest inner loop
    {
        "mode": "full",
        "inner_steps": 4,
        "inner_lr": 2e-2,
        "inner_momentum": 0.95,
        "modifier_expansion": 4,
        "surprise_type": "full",
        "persist_momentum": True,
        "residual_init": 0.08,
    },
]


class CMSSelfModifyingEncoder(nn.Module):
    """Encoder wrapper tuân thủ CMS principle.

    Args:
        backbone: pre-trained encoder (tương thích với build_encoder)
        feature_channels: list 4 phần tử [c2, c3, c4, c5]
        stage_configs: list 4 dict per-stage (xem DEFAULT_STAGE_CONFIG)
                       nếu None → dùng DEFAULT_STAGE_CONFIG
        backbone_lr_decay: LLRD factor (0.5 = shallow chậm gấp 2 deep)
        unfreeze_schedule: dict điều khiển progressive unfreeze
                          {"deep": 0, "mid": 8, "shallow": 16}
                          nghĩa là: từ epoch 0 unfreeze c4,c5;
                                    từ epoch 8 unfreeze c3;
                                    từ epoch 16 unfreeze c2 + stem.
                          Set tất cả = 0 để disable progressive (unfreeze ngay).
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_channels: List[int] = [64, 128, 320, 512],
        stage_configs: Optional[List[Dict[str, Any]]] = None,
        backbone_lr_decay: float = 0.5,
        unfreeze_schedule: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        assert len(feature_channels) == 4, "Expect 4 feature channels (c2..c5)"

        self.backbone = backbone
        self.feature_channels = list(feature_channels)
        self.backbone_lr_decay = float(backbone_lr_decay)

        if stage_configs is None:
            stage_configs = DEFAULT_STAGE_CONFIG
        assert len(stage_configs) == 4
        self.stage_configs = list(stage_configs)

        if unfreeze_schedule is None:
            unfreeze_schedule = {"deep": 0, "mid": 8, "shallow": 16}
        self.unfreeze_schedule = dict(unfreeze_schedule)
        self._current_epoch = 0

        # Tạo adaptor/modifier per-stage theo config
        self.stage_modules = nn.ModuleDict()
        self.pre_norms = nn.ModuleDict()
        self.stage_modes: List[str] = []

        for idx, (ch, cfg) in enumerate(zip(feature_channels, stage_configs)):
            mode = cfg.get("mode", "none")
            self.stage_modes.append(mode)
            key = f"stage_{idx}"
            if mode == "none":
                continue
            # Pre-norm để stabilize input cho adaptor/modifier
            self.pre_norms[key] = nn.GroupNorm(min(8, max(1, ch // 4)), ch)
            if mode == "light":
                self.stage_modules[key] = LightAdaptor(
                    channels=ch,
                    rank=int(cfg.get("lora_rank", 4)),
                    residual_init=float(cfg.get("residual_init", 0.05)),
                )
            elif mode == "full":
                self.stage_modules[key] = CMSSelfModifyingBlock(
                    channels=ch,
                    inner_steps=int(cfg.get("inner_steps", 2)),
                    inner_lr=float(cfg.get("inner_lr", 5e-3)),
                    inner_momentum=float(cfg.get("inner_momentum", 0.9)),
                    modifier_expansion=int(cfg.get("modifier_expansion", 2)),
                    dropout=float(cfg.get("dropout", 0.1)),
                    surprise_type=str(cfg.get("surprise_type", "full")),
                    persist_momentum=bool(cfg.get("persist_momentum", True)),
                    residual_init=float(cfg.get("residual_init", 0.05)),
                )
            else:
                raise ValueError(f"Unknown stage mode: {mode}")

        # Apply initial freeze state
        self.set_epoch(0)

    # ------------------------------------------------------------------
    # Progressive unfreeze control
    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Gọi từ training loop mỗi epoch để cập nhật trạng thái freeze."""
        self._current_epoch = int(epoch)
        self._apply_freeze_policy()

    def _apply_freeze_policy(self) -> None:
        epoch = self._current_epoch
        schedule = self.unfreeze_schedule
        # Mặc định freeze toàn bộ backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Quy tắc:
        # - epoch >= schedule["deep"]     → unfreeze c4, c5 của backbone
        # - epoch >= schedule["mid"]      → unfreeze c3
        # - epoch >= schedule["shallow"]  → unfreeze c2 + stem
        #
        # Cách nhận biết "stage nào là stage nào" trong backbone khá phụ
        # thuộc implementation. Chiến lược an toàn: ta đánh dấu parameter
        # theo thứ tự xuất hiện (earlier = shallower) rồi chia thành 4
        # nhóm gần bằng nhau + 1 nhóm stem.
        param_list = list(self.backbone.named_parameters())
        if not param_list:
            return

        n = len(param_list)
        # 5 nhóm: stem(10%), c2(20%), c3(25%), c4(25%), c5(20%)
        boundaries = [
            int(n * 0.10),  # stem end
            int(n * 0.30),  # c2 end
            int(n * 0.55),  # c3 end
            int(n * 0.80),  # c4 end
        ]

        def unfreeze_range(start: int, end: int):
            for i in range(start, end):
                _, p = param_list[i]
                p.requires_grad = True

        # Deep (c4, c5) — unfreeze trước
        if epoch >= schedule.get("deep", 0):
            unfreeze_range(boundaries[2], n)            # c4 + c5
        # Mid (c3)
        if epoch >= schedule.get("mid", 8):
            unfreeze_range(boundaries[1], boundaries[2])  # c3
        # Shallow (c2 + stem)
        if epoch >= schedule.get("shallow", 16):
            unfreeze_range(0, boundaries[1])              # stem + c2

    # ------------------------------------------------------------------
    # Parameter groups cho optimizer (LLRD)
    # ------------------------------------------------------------------
    def build_param_groups(
        self,
        base_backbone_lr: float,
        adaptor_lr: float,
        modifier_inner_lr_scale: float = 0.5,
        weight_decay: float = 1e-4,
    ) -> List[Dict[str, Any]]:
        """Trả list param_groups dùng trực tiếp cho AdamW.

        LLRD ý tưởng:
            stem    : base_backbone_lr * decay^4
            c2 block: base_backbone_lr * decay^3
            c3 block: base_backbone_lr * decay^2
            c4 block: base_backbone_lr * decay^1
            c5 block: base_backbone_lr * decay^0

        adaptor_lr dùng cho modifier/adaptor outer params (gate, residual_scale,
        reconstructor, pre_norm, up/down của LoRA, v.v.). Inner loop params
        (self.modifier.*.weight trong CMSSelfModifyingBlock) cũng cần outer
        gradient (vì final_modified vẫn nằm trong graph), dùng LR = adaptor_lr
        * modifier_inner_lr_scale.
        """
        groups: List[Dict[str, Any]] = []
        decay = self.backbone_lr_decay

        # --- Backbone: 5 nhóm với LR giảm dần từ deep về shallow ---
        param_list = list(self.backbone.named_parameters())
        n = len(param_list)
        if n > 0:
            boundaries = [
                int(n * 0.10),
                int(n * 0.30),
                int(n * 0.55),
                int(n * 0.80),
            ]
            ranges = [
                ("backbone_stem", 0, boundaries[0], decay ** 4),
                ("backbone_c2", boundaries[0], boundaries[1], decay ** 3),
                ("backbone_c3", boundaries[1], boundaries[2], decay ** 2),
                ("backbone_c4", boundaries[2], boundaries[3], decay ** 1),
                ("backbone_c5", boundaries[3], n, decay ** 0),
            ]
            for name, start, end, scale in ranges:
                params = [p for _, p in param_list[start:end] if p.requires_grad or True]
                # requires_grad được set per-epoch, nhưng optimizer state
                # tạo một lần — ta vẫn đưa vào group, param có grad=None thì
                # AdamW sẽ skip step một cách an toàn.
                if params:
                    groups.append({
                        "params": params,
                        "lr": base_backbone_lr * scale,
                        "weight_decay": weight_decay,
                        "name": name,
                    })

        # --- Adaptor / modifier outer params ---
        for key, module in self.stage_modules.items():
            outer_params: List[nn.Parameter] = []
            inner_params: List[nn.Parameter] = []
            for pname, p in module.named_parameters():
                # Inner loop target = self.modifier.* (trong CMSSelfModifyingBlock)
                # → LR nhỏ hơn vì đã được inner SGD update rồi
                if isinstance(module, CMSSelfModifyingBlock) and pname.startswith("modifier."):
                    inner_params.append(p)
                else:
                    outer_params.append(p)
            if outer_params:
                groups.append({
                    "params": outer_params,
                    "lr": adaptor_lr,
                    "weight_decay": weight_decay,
                    "name": f"adaptor_outer_{key}",
                })
            if inner_params:
                groups.append({
                    "params": inner_params,
                    "lr": adaptor_lr * modifier_inner_lr_scale,
                    "weight_decay": weight_decay * 0.1,  # nhẹ hơn cho inner
                    "name": f"modifier_inner_{key}",
                })

        # --- Pre-norms ---
        pn_params = list(self.pre_norms.parameters())
        if pn_params:
            groups.append({
                "params": pn_params,
                "lr": adaptor_lr,
                "weight_decay": 0.0,  # GroupNorm không nên weight-decay
                "name": "pre_norms",
            })

        return groups

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, x: Tensor, return_nested_info: bool = False,
    ) -> Tuple[List[Tensor], Dict[str, Dict]]:
        features = self.backbone(x)
        if isinstance(features, (tuple, list)):
            features = list(features)
        else:
            raise ValueError("Backbone must return list/tuple of 4 feature maps")
        if len(features) != 4:
            raise RuntimeError(f"Expected 4 feature maps, got {len(features)}")

        nested_info: Dict[str, Dict] = {}

        for idx, mode in enumerate(self.stage_modes):
            if mode == "none":
                continue
            key = f"stage_{idx}"
            feat = self.pre_norms[key](features[idx])
            modified_feat, info = self.stage_modules[key](
                feat, return_info=return_nested_info
            )
            features[idx] = modified_feat
            if return_nested_info and info is not None:
                info["stage_idx"] = idx
                info["mode"] = mode
                nested_info[key] = info

        return features, nested_info

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def describe(self) -> str:
        lines = [f"CMSSelfModifyingEncoder (epoch={self._current_epoch})"]
        lines.append(f"  backbone_lr_decay = {self.backbone_lr_decay}")
        lines.append(f"  unfreeze_schedule = {self.unfreeze_schedule}")
        for idx, (mode, cfg) in enumerate(zip(self.stage_modes, self.stage_configs)):
            stage_name = f"c{idx + 2}"
            lines.append(f"  {stage_name} [{mode}] :: {cfg}")
        # Đếm params requires_grad
        bb_train = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        bb_total = sum(p.numel() for p in self.backbone.parameters())
        mod_train = sum(
            p.numel()
            for p in list(self.stage_modules.parameters()) + list(self.pre_norms.parameters())
            if p.requires_grad
        )
        lines.append(
            f"  backbone trainable: {bb_train:,} / {bb_total:,} "
            f"({100 * bb_train / max(bb_total, 1):.1f}%)"
        )
        lines.append(f"  adaptor+modifier trainable: {mod_train:,}")
        return "\n".join(lines)


# =============================================================================
# 5. SMOKE TESTS
# =============================================================================


def _smoke_test():
    print("=" * 70)
    print("CMSSelfModifyingEncoder — smoke tests")
    print("=" * 70)

    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, 64, 3, padding=1)
            self.b1 = nn.Conv2d(64, 64, 3, padding=1)
            self.b2 = nn.Conv2d(64, 128, 3, padding=1)
            self.b3 = nn.Conv2d(128, 320, 3, padding=1)
            self.b4 = nn.Conv2d(320, 512, 3, padding=1)
            self.out_channels = [64, 128, 320, 512]

        def forward(self, x):
            B = x.shape[0]
            return [
                torch.randn(B, 64, 96, 96, device=x.device),
                torch.randn(B, 128, 48, 48, device=x.device),
                torch.randn(B, 320, 24, 24, device=x.device),
                torch.randn(B, 512, 12, 12, device=x.device),
            ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = DummyBackbone().to(device)
    encoder = CMSSelfModifyingEncoder(
        backbone=backbone,
        feature_channels=[64, 128, 320, 512],
        backbone_lr_decay=0.5,
        unfreeze_schedule={"deep": 0, "mid": 8, "shallow": 16},
    ).to(device)

    print("\n[1] describe() epoch=0:")
    print(encoder.describe())

    print("\n[2] Forward pass + gradient flow:")
    x = torch.randn(2, 3, 384, 384, device=device)
    encoder.train()
    features, info = encoder(x, return_nested_info=True)
    for i, f in enumerate(features):
        print(f"    c{i+2}: {tuple(f.shape)}")
    print(f"    nested_info stages: {list(info.keys())}")
    for key, stage_info in info.items():
        print(
            f"      {key} [{stage_info['mode']}]: "
            f"gate={stage_info['gate_value']:.3f}, "
            f"scale={stage_info['residual_scale']:.3f}, "
            f"steps={len(stage_info.get('inner_steps', []))}, "
            f"persist={stage_info['persist_momentum']}"
        )

    loss = sum(f.mean() for f in features) + 0.1 * sum(
        info[k]["aux_surprise"] for k in info if "aux_surprise" in info[k]
    )
    loss.backward()
    print(f"    loss.backward() OK, loss={loss.item():.4f}")

    print("\n[3] Progressive unfreeze check:")
    for ep in [0, 5, 10, 20]:
        encoder.set_epoch(ep)
        bb_train = sum(p.numel() for p in encoder.backbone.parameters() if p.requires_grad)
        bb_total = sum(p.numel() for p in encoder.backbone.parameters())
        pct = 100 * bb_train / max(bb_total, 1)
        print(f"    epoch={ep:>3}: backbone trainable = {bb_train:,} / {bb_total:,} ({pct:.1f}%)")

    print("\n[4] Param groups for optimizer (LLRD):")
    encoder.set_epoch(20)  # unfreeze all
    groups = encoder.build_param_groups(
        base_backbone_lr=1e-4,
        adaptor_lr=3e-4,
        modifier_inner_lr_scale=0.5,
        weight_decay=1e-4,
    )
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        print(f"    {g['name']:<28} lr={g['lr']:.2e}  wd={g['weight_decay']:.1e}  params={n:,}")

    print("\n[5] Persistent momentum buffer (c5 deep stage):")
    key = "stage_3"
    block = encoder.stage_modules[key]
    if isinstance(block, CMSSelfModifyingBlock) and block.persist_momentum:
        # run 2 forwards, momentum buffer phải accumulate
        encoder.train()
        _ = encoder(x, return_nested_info=False)
        buf1 = list(block._momentum_buffer.values())[0].clone() if block._momentum_buffer else None
        _ = encoder(x, return_nested_info=False)
        buf2 = list(block._momentum_buffer.values())[0].clone() if block._momentum_buffer else None
        if buf1 is not None and buf2 is not None:
            diff = (buf2 - buf1).abs().mean().item()
            print(f"    buf1.mean={buf1.abs().mean():.4e}")
            print(f"    buf2.mean={buf2.abs().mean():.4e}")
            print(f"    |buf2-buf1|.mean={diff:.4e} (should be > 0)")

    print("\n" + "=" * 70)
    print("All smoke tests finished")
    print("=" * 70)


if __name__ == "__main__":
    _smoke_test()
