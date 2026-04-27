"""
Minimal CMS-ordering patch cho SelfModifyingEncoder
====================================================

Muc tieu: giu DUNG mot thay doi ma paradigm CMS doi hoi -
   "deeper stage = more inner updates, stronger self-modification"
Khong dung vao bat ky thu gi khac (khong LLRD, khong progressive
unfreeze, khong LoRA them, khong persistent momentum).

Thay doi duy nhat so voi `self_modifying_encoder.py` cu:

1. Bo cong thuc `stage_lr = inner_lr * (0.5 ** stage_idx)` - cong thuc
   nay lam inner_lr GIAM khi stage sau hon (c5 = 0.00125 voi base=0.01),
   tuc NGUOC voi CMS.
2. Thay bang `inner_lr_schedule: List[float]` per-stage, mac dinh
   tang dan theo depth.
3. `inner_steps_schedule` giu nhu cu (da tang dan san).

Moi thu con lai (module tree, forward, inner loop reset-per-sample,
pre_norms, surprise objective) copy nguyen tu ban cu.

Ablation flags them vao (tat ca DEFAULT OFF de khong dong ket qua):
    - `use_llrd`              : layer-wise LR decay tren backbone
    - `use_progressive_unfreeze`: freeze dan
    - `c3_adaptor_mode`       : "none" (default) | "light"
    - `use_persistent_momentum`: momentum buffer giua cac batch

Cach dung:

    # Buoc 1: chi thu CMS ordering (thay doi duy nhat)
    encoder = MinimalCMSEncoder(
        backbone=backbone,
        feature_channels=[64, 128, 320, 512],
        apply_stages=[2, 3],           # y het ban cu
        inner_steps_schedule=[1, 2, 3, 4],    # giu nhu cu
        inner_lr_schedule=[0, 0, 0.005, 0.015],  # MOI: tang dan
    )
    # Neu ket qua giam: ordering khong phai thu pham.
    # Neu ket qua tang: dung la user muon dieu nay.

    # Buoc 2: bat them tung feature mot
    encoder = MinimalCMSEncoder(
        ...,
        use_llrd=True,                 # test rieng LLRD
    )
    # v.v.

Interface: gan nhu drop-in thay cho `SelfModifyingEncoder` cu.
Ham forward(x, return_nested_info=True) giong het.
"""

import copy
import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Surprise (identical to original)
# =============================================================================


class SurpriseObjective(nn.Module):
    def __init__(self, channels: int, spatial_mask_ratio: float = 0.25):
        super().__init__()
        self.spatial_mask_ratio = spatial_mask_ratio
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

    def forward(self, x: Tensor):
        _, C, _, _ = x.shape
        mask = self._spatial_mask(x)
        reconstructed = self.reconstructor(x * mask)
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
                self.running_mean.lerp_(current_mean.detach().to(dtype=buf_dtype), self.ema_momentum)
                self.running_var.lerp_(current_var.detach().to(dtype=buf_dtype), self.ema_momentum)
        rm = self.running_mean.to(dtype=current_mean.dtype)
        rv = self.running_var.to(dtype=current_var.dtype)
        consist_loss = (
            (current_mean - rm).pow(2).mean()
            + (current_var / (rv + 1e-8) - 1.0).pow(2).mean()
        )
        total = recon_loss + 0.1 * consist_loss
        return total, {
            "recon_loss": recon_loss.detach(),
            "consist_loss": consist_loss.detach(),
            "total_surprise": total.detach(),
        }


# =============================================================================
# SelfModifyingBlock - 95% copy cua ban cu, chi them flag persist_momentum
# =============================================================================


class SelfModifyingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        inner_steps: int = 3,
        inner_lr: float = 0.01,
        modifier_expansion: int = 2,
        dropout: float = 0.1,
        # ABLATION flag - mac dinh False de matching ban cu
        persist_momentum: bool = False,
        inner_momentum_beta: float = 0.9,
    ):
        super().__init__()
        self.channels = channels
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.persist_momentum = persist_momentum
        self.inner_momentum_beta = inner_momentum_beta

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
        self.surprise = SurpriseObjective(channels)
        self.residual_scale = nn.Parameter(torch.tensor(0.05))
        self.register_buffer("log_inner_lr", torch.tensor(math.log(inner_lr)))
        self._momentum_buffer: Optional[Dict[str, Tensor]] = None

    def _get_inner_lr(self) -> float:
        return self.log_inner_lr.exp().item()

    def _ensure_momentum_buffer(self):
        if self.persist_momentum and self._momentum_buffer is None:
            self._momentum_buffer = {
                name: torch.zeros_like(p.data)
                for name, p in self.modifier.named_parameters()
            }

    def _inner_loop(self, features: Tensor):
        inner_lr = self._get_inner_lr()
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
                    mod_params = [(n, p) for n, p in self.modifier.named_parameters() if p.requires_grad]
                    if not mod_params or not surprise_loss.requires_grad:
                        continue
                    grads = torch.autograd.grad(
                        surprise_loss, [p for _, p in mod_params],
                        create_graph=False, retain_graph=False, allow_unused=True,
                    )
                    with torch.no_grad():
                        for (name, p), g in zip(mod_params, grads):
                            if g is None:
                                continue
                            if self.persist_momentum:
                                buf = self._momentum_buffer[name]
                                buf.mul_(self.inner_momentum_beta).add_(g)
                                p.data.sub_(inner_lr * buf)
                            else:
                                p.data.sub_(inner_lr * g)

        final_modified = self.modifier(features)
        if any(p.requires_grad for p in self.modifier.parameters()):
            aux_surprise, _ = self.surprise(final_modified + features)
        else:
            aux_surprise = torch.tensor(0.0, device=features.device)
        return final_modified, aux_surprise, all_info

    def forward(self, x: Tensor, return_info: bool = False):
        saved_state = {n: p.data.clone() for n, p in self.modifier.named_parameters()}
        modification, aux_surprise, steps_info = self._inner_loop(x)
        gate = self.gate(x).unsqueeze(-1).unsqueeze(-1)
        output = x + self.residual_scale * gate * modification
        for n, p in self.modifier.named_parameters():
            p.data.copy_(saved_state[n])
        info = None
        if return_info:
            info = {
                "gate_value": float(gate.detach().mean().item()),
                "residual_scale": float(self.residual_scale.item()),
                "inner_lr": self._get_inner_lr(),
                "inner_steps": steps_info,
                "aux_surprise": aux_surprise,
            }
        return output, info


# =============================================================================
# LightAdaptor - chi dung khi c3_adaptor_mode="light" (ablation)
# =============================================================================


class LightAdaptor(nn.Module):
    def __init__(self, channels: int, rank: int = 4):
        super().__init__()
        self.down = nn.Conv2d(channels, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, channels, 1, bias=False)
        nn.init.kaiming_normal_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)  # init 0 - delta = 0 luc dau
        self.residual_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x: Tensor, return_info: bool = False):
        delta = self.up(F.gelu(self.down(x)))
        out = x + self.residual_scale * delta
        if return_info:
            return out, {
                "gate_value": 1.0, "residual_scale": float(self.residual_scale.item()),
                "inner_lr": 0.0, "inner_steps": [],
                "aux_surprise": torch.zeros((), device=x.device, dtype=x.dtype),
            }
        return out, None


# =============================================================================
# MinimalCMSEncoder - drop-in thay cho SelfModifyingEncoder cu
# =============================================================================


class MinimalCMSEncoder(nn.Module):
    """Minimal CMS ordering patch.

    Thay doi ACTIVE (luon bat):
      1. `inner_lr_schedule` per-stage (thay `inner_lr * 0.5^stage_idx`)
         Default: [0, 0, 0.005, 0.015] - tang dan voi depth

    Ablation flags (DEFAULT OFF, bat tung cai de test):
      2. `use_llrd=True`               - LLRD tren backbone
      3. `use_progressive_unfreeze=True`- freeze dan (can goi set_epoch)
      4. `c3_adaptor_mode="light"`     - them LoRA rank-4 len c3
      5. `use_persistent_momentum=True`- momentum buffer qua cac batch

    Voi tat ca flag OFF, module nay CHI khac ban cu o inner_lr_schedule.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_channels: List[int] = [64, 128, 320, 512],
        # --- Core (active) ---
        apply_stages: Optional[List[int]] = None,
        inner_steps_schedule: Optional[List[int]] = None,
        inner_lr_schedule: Optional[List[float]] = None,
        modifier_expansion: int = 2,
        dropout: float = 0.2,
        # --- Ablation flags (default matching ban cu) ---
        use_llrd: bool = False,
        backbone_lr_decay: float = 0.65,
        use_progressive_unfreeze: bool = False,
        unfreeze_schedule: Optional[Dict[str, int]] = None,
        c3_adaptor_mode: str = "none",  # "none" | "light"
        c3_lora_rank: int = 4,
        use_persistent_momentum: bool = False,
        inner_momentum_beta: float = 0.9,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_channels = list(feature_channels)

        # Defaults
        if apply_stages is None:
            apply_stages = [2, 3]
        if inner_steps_schedule is None:
            inner_steps_schedule = [1, 2, 3, 4]
        if inner_lr_schedule is None:
            # tang dan theo depth; stage khong trong apply_stages thi 0
            inner_lr_schedule = [0.005, 0.015, 0.02, 0.05]

        assert len(inner_steps_schedule) == 4
        assert len(inner_lr_schedule) == 4

        self.apply_stages = list(apply_stages)
        self.inner_steps_schedule = list(inner_steps_schedule)
        self.inner_lr_schedule = list(inner_lr_schedule)

        # Ablation state
        self.use_llrd = use_llrd
        self.backbone_lr_decay = backbone_lr_decay
        self.use_progressive_unfreeze = use_progressive_unfreeze
        self.unfreeze_schedule = unfreeze_schedule or {"deep": 0, "mid": 10, "shallow": 20}
        self.c3_adaptor_mode = c3_adaptor_mode

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Build modifiers & pre_norms
        self.modifiers = nn.ModuleDict()
        self.pre_norms = nn.ModuleDict()

        for stage_idx in self.apply_stages:
            ch = feature_channels[stage_idx]
            key = f"stage_{stage_idx}"
            self.pre_norms[key] = nn.GroupNorm(min(8, max(1, ch // 4)), ch)
            self.modifiers[key] = SelfModifyingBlock(
                channels=ch,
                inner_steps=inner_steps_schedule[stage_idx],
                inner_lr=inner_lr_schedule[stage_idx],
                modifier_expansion=modifier_expansion,
                dropout=dropout,
                persist_momentum=use_persistent_momentum,
                inner_momentum_beta=inner_momentum_beta,
            )

        # Optional c3 adaptor (ablation)
        self.c3_adaptor: Optional[nn.Module] = None
        if c3_adaptor_mode == "light":
            self.c3_adaptor = LightAdaptor(feature_channels[1], rank=c3_lora_rank)
            if "stage_1" not in self.pre_norms:
                self.pre_norms["stage_1"] = nn.GroupNorm(
                    min(8, max(1, feature_channels[1] // 4)), feature_channels[1]
                )

        # Progressive unfreeze init
        self._current_epoch = 0
        if use_progressive_unfreeze:
            self.set_epoch(0)

    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int):
        self._current_epoch = int(epoch)
        if not self.use_progressive_unfreeze:
            return
        for p in self.backbone.parameters():
            p.requires_grad = False
        param_list = list(self.backbone.named_parameters())
        n = len(param_list)
        if n == 0:
            return
        b = [int(n * 0.10), int(n * 0.30), int(n * 0.55), int(n * 0.80)]

        def unfreeze_range(start, end):
            for i in range(start, end):
                param_list[i][1].requires_grad = True

        if epoch >= self.unfreeze_schedule.get("deep", 0):
            unfreeze_range(b[2], n)
        if epoch >= self.unfreeze_schedule.get("mid", 10):
            unfreeze_range(b[1], b[2])
        if epoch >= self.unfreeze_schedule.get("shallow", 20):
            unfreeze_range(0, b[1])

    # ------------------------------------------------------------------
    def build_param_groups(
        self,
        base_backbone_lr: float,
        adaptor_lr: float,
        modifier_inner_lr_scale: float = 0.5,  # ignored, kept for API compat
        weight_decay: float = 1e-4,
    ) -> List[Dict[str, Any]]:
        """Returns LLRD groups if `use_llrd=True`, else single group."""
        groups: List[Dict[str, Any]] = []
        param_list = list(self.backbone.named_parameters())
        n = len(param_list)

        if self.use_llrd and n > 0:
            b = [int(n * 0.10), int(n * 0.30), int(n * 0.55), int(n * 0.80)]
            d = self.backbone_lr_decay
            ranges = [
                ("backbone_stem", 0,    b[0], d ** 4),
                ("backbone_c2",   b[0], b[1], d ** 3),
                ("backbone_c3",   b[1], b[2], d ** 2),
                ("backbone_c4",   b[2], b[3], d ** 1),
                ("backbone_c5",   b[3], n,    1.0),
            ]
            for name, s, e, scale in ranges:
                ps = [p for _, p in param_list[s:e]]
                if ps:
                    groups.append({
                        "params": ps,
                        "lr": base_backbone_lr * scale,
                        "weight_decay": weight_decay,
                        "name": name,
                    })
        else:
            bb_params = [p for _, p in param_list]
            if bb_params:
                groups.append({
                    "params": bb_params,
                    "lr": base_backbone_lr,
                    "weight_decay": weight_decay,
                    "name": "backbone",
                })

        adaptor_params = (
            list(self.modifiers.parameters())
            + list(self.pre_norms.parameters())
        )
        if self.c3_adaptor is not None:
            adaptor_params += list(self.c3_adaptor.parameters())
        if adaptor_params:
            groups.append({
                "params": adaptor_params,
                "lr": adaptor_lr,
                "weight_decay": weight_decay,
                "name": "adaptor",
            })
        return groups

    # ------------------------------------------------------------------
    def forward(self, x: Tensor, return_nested_info: bool = False):
        features = self.backbone(x)
        features = list(features) if isinstance(features, (tuple, list)) else [features]
        if len(features) != 4:
            raise RuntimeError(f"Expected 4 feature maps, got {len(features)}")

        nested_info: Dict[str, Dict] = {}

        # Apply c3 adaptor first (if enabled)
        if self.c3_adaptor is not None:
            feat = self.pre_norms["stage_1"](features[1])
            features[1], info = self.c3_adaptor(feat, return_info=return_nested_info)
            if return_nested_info and info is not None:
                nested_info["stage_1"] = info

        for stage_idx in self.apply_stages:
            key = f"stage_{stage_idx}"
            if key not in self.modifiers:
                continue
            feat = self.pre_norms[key](features[stage_idx])
            features[stage_idx], info = self.modifiers[key](
                feat, return_info=return_nested_info
            )
            if return_nested_info and info is not None:
                nested_info[key] = info

        return features, nested_info

    # ------------------------------------------------------------------
    def describe(self) -> str:
        lines = [f"MinimalCMSEncoder (epoch={self._current_epoch})"]
        lines.append(f"  apply_stages        = {self.apply_stages}")
        lines.append(f"  inner_steps_schedule= {self.inner_steps_schedule}")
        lines.append(f"  inner_lr_schedule   = {self.inner_lr_schedule}")
        lines.append(f"  use_llrd            = {self.use_llrd}")
        lines.append(f"  use_progressive_unfreeze = {self.use_progressive_unfreeze}")
        lines.append(f"  c3_adaptor_mode     = {self.c3_adaptor_mode}")
        lines.append(f"  use_persistent_momentum = "
                     f"{any(b.persist_momentum for b in self.modifiers.values()) if self.modifiers else False}")
        return "\n".join(lines)


# =============================================================================
# Smoke test
# =============================================================================


def _smoke_test():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
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
    x = torch.randn(2, 3, 384, 384, device=device)

    print("=" * 60)
    print("[1] Config A: minimal - CHI CMS ordering change")
    print("=" * 60)
    enc = MinimalCMSEncoder(Dummy().to(device)).to(device)
    enc.train()
    print(enc.describe())
    feats, info = enc(x, return_nested_info=True)
    print(f"  stages modified: {list(info.keys())}")
    for k, v in info.items():
        print(f"    {k}: inner_lr={v['inner_lr']:.4f}, steps={len(v.get('inner_steps', []))}")
    loss = sum(f.mean() for f in feats)
    loss.backward()
    print(f"  backward OK, loss={loss.item():.4f}\n")

    print("=" * 60)
    print("[2] Config B: minimal + LLRD only")
    print("=" * 60)
    enc = MinimalCMSEncoder(Dummy().to(device), use_llrd=True, backbone_lr_decay=0.65).to(device)
    groups = enc.build_param_groups(1e-4, 3e-4)
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:<20} lr={g['lr']:.2e}  params={n:,}")
    print()

    print("=" * 60)
    print("[3] Config C: minimal + c3 adaptor only")
    print("=" * 60)
    enc = MinimalCMSEncoder(Dummy().to(device), c3_adaptor_mode="light").to(device)
    feats, info = enc(x, return_nested_info=True)
    print(f"  stages modified: {list(info.keys())}")
    print()

    print("All configs run OK. Now toggle flags one at a time in training.")


if __name__ == "__main__":
    _smoke_test()
