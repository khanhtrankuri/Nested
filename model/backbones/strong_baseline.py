import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma[:, None, None] * x
        if self.training and self.drop_path > 0.0:
            keep = 1.0 - self.drop_path
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            x = x.div(keep) * random_tensor
        return residual + x


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyConvNeXtEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, dims: List[int] = [64, 128, 256, 512], depths: List[int] = [2, 2, 3, 2]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )

        self.stage1 = nn.Sequential(*[ConvNeXtBlock(dims[0], drop_path=0.0) for _ in range(depths[0])])
        self.down1 = Downsample(dims[0], dims[1])
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(dims[1], drop_path=0.02) for _ in range(depths[1])])
        self.down2 = Downsample(dims[1], dims[2])
        self.stage3 = nn.Sequential(*[ConvNeXtBlock(dims[2], drop_path=0.04) for _ in range(depths[2])])
        self.down3 = Downsample(dims[2], dims[3])
        self.stage4 = nn.Sequential(*[ConvNeXtBlock(dims[3], drop_path=0.06) for _ in range(depths[3])])
        self.out_channels = dims

    def forward(self, x: torch.Tensor):
        c2 = self.stage1(self.stem(x))   # 1/4
        c3 = self.stage2(self.down1(c2)) # 1/8
        c4 = self.stage3(self.down2(c3)) # 1/16
        c5 = self.stage4(self.down3(c4)) # 1/32
        return c2, c3, c4, c5


class TorchvisionConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        variant: str = "convnext_tiny",
        in_channels: int = 3,
        use_pretrained: bool = False,
        strict_pretrained: bool = False,
        pretrained_cache_dir: Optional[str] = None,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"{variant} only supports in_channels=3, got {in_channels}.")

        try:
            from torchvision.models import (
                ConvNeXt_Small_Weights,
                ConvNeXt_Tiny_Weights,
                convnext_small,
                convnext_tiny,
            )
        except ImportError as exc:
            raise ImportError("Torchvision ConvNeXt encoder requires torchvision to be installed.") from exc

        builders = {
            "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
            "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT),
        }
        if variant not in builders:
            raise ValueError(f"Unsupported ConvNeXt variant: {variant}")

        builder, default_weights = builders[variant]
        loaded_pretrained = False
        try:
            old_torch_hub_dir = None
            if pretrained_cache_dir is not None:
                os.makedirs(pretrained_cache_dir, exist_ok=True)
                old_torch_hub_dir = torch.hub.get_dir()
                torch.hub.set_dir(pretrained_cache_dir)
            backbone = builder(weights=default_weights if use_pretrained else None)
            loaded_pretrained = bool(use_pretrained)
        except Exception as exc:
            if not use_pretrained or strict_pretrained:
                raise
            print(
                f"[StrongBaseline] Failed to load pretrained weights for {variant}: {exc}. "
                "Falling back to random initialization."
            )
            backbone = builder(weights=None)
        finally:
            if pretrained_cache_dir is not None and old_torch_hub_dir is not None:
                torch.hub.set_dir(old_torch_hub_dir)

        self.features = backbone.features
        self.out_channels = [96, 192, 384, 768]
        self.variant = variant
        self.pretrained_loaded = loaded_pretrained

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.features[0](x)
        c2 = self.features[1](x)
        x = self.features[2](c2)
        c3 = self.features[3](x)
        x = self.features[4](c3)
        c4 = self.features[5](x)
        x = self.features[6](c4)
        c5 = self.features[7](x)
        return c2, c3, c4, c5


def build_encoder(
    encoder_name: str,
    in_channels: int = 3,
    use_pretrained: bool = False,
    strict_pretrained: bool = False,
    pretrained_cache_dir: Optional[str] = None,
):
    encoder_name = encoder_name.lower()
    if encoder_name == "tiny_convnext":
        encoder = TinyConvNeXtEncoder(in_channels=in_channels)
        encoder.pretrained_loaded = False
        return encoder
    if encoder_name in {"convnext_tiny", "convnext_small"}:
        return TorchvisionConvNeXtEncoder(
            variant=encoder_name,
            in_channels=in_channels,
            use_pretrained=use_pretrained,
            strict_pretrained=strict_pretrained,
            pretrained_cache_dir=pretrained_cache_dir,
        )
    raise ValueError(f"Unsupported encoder_name: {encoder_name}")


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FPNDecoder(nn.Module):
    def __init__(self, encoder_channels, pyramid_channels: int = 256, seg_channels: int = 128):
        super().__init__()
        c2, c3, c4, c5 = encoder_channels
        self.lateral5 = nn.Conv2d(c5, pyramid_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(c4, pyramid_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(c3, pyramid_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(c2, pyramid_channels, kernel_size=1)
        self.smooth5 = ConvBNAct(pyramid_channels, seg_channels)
        self.smooth4 = ConvBNAct(pyramid_channels, seg_channels)
        self.smooth3 = ConvBNAct(pyramid_channels, seg_channels)
        self.smooth2 = ConvBNAct(pyramid_channels, seg_channels)
        self.fuse = nn.Sequential(ConvBNAct(seg_channels * 4, seg_channels), ConvBNAct(seg_channels, seg_channels))

    def forward(self, c2, c3, c4, c5):
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        s5 = self.smooth5(p5)
        s4 = self.smooth4(p4)
        s3 = self.smooth3(p3)
        s2 = self.smooth2(p2)
        target_size = s2.shape[-2:]
        s3 = F.interpolate(s3, size=target_size, mode="bilinear", align_corners=False)
        s4 = F.interpolate(s4, size=target_size, mode="bilinear", align_corners=False)
        s5 = F.interpolate(s5, size=target_size, mode="bilinear", align_corners=False)
        fused = torch.cat([s2, s3, s4, s5], dim=1)
        return self.fuse(fused), s5


class SafeNestedResidualRefiner(nn.Module):
    def __init__(
        self,
        feat_channels: int,
        nested_dim: int = 128,
        num_prototypes: int = 8,
        residual_scale: float = 0.05,
        prototype_max_norm: float = 1.0,
        memory_hidden_dim: int = 128,
        slow_momentum_scale: float = 0.25,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.nested_dim = nested_dim
        self.num_prototypes = num_prototypes
        self.residual_scale = float(residual_scale)
        self.prototype_max_norm = float(prototype_max_norm)
        self.slow_momentum_scale = float(slow_momentum_scale)

        self.query_proj = nn.Conv2d(feat_channels, nested_dim, kernel_size=1, bias=False)
        self.residual_head = nn.Sequential(
            ConvBNAct(feat_channels + nested_dim + 1, feat_channels),
            nn.Conv2d(feat_channels, 1, kernel_size=1),
        )
        gate_in_dim = nested_dim + 4
        hidden_dim = max(int(memory_hidden_dim), 32)
        self.memory_gate = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        self.register_buffer("fast_prototypes", torch.randn(num_prototypes, nested_dim) * init_std)
        self.register_buffer("slow_prototypes", torch.randn(num_prototypes, nested_dim) * init_std)

    def _uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        return 1.0 - torch.abs(2.0 * probs - 1.0)

    def _compute_token(self, query_feat: torch.Tensor, coarse_lowres_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(coarse_lowres_logits)
        uncertainty = self._uncertainty(coarse_lowres_logits)
        focus = 0.70 * probs + 0.30 * uncertainty
        denom = focus.sum(dim=(2, 3), keepdim=True).clamp(min=1e-5)
        token = (query_feat * focus).sum(dim=(2, 3), keepdim=True) / denom
        token = token.flatten(1)
        return F.normalize(token, dim=-1)

    def _memory_stats(self, coarse_lowres_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(coarse_lowres_logits)
        uncertainty = self._uncertainty(coarse_lowres_logits)
        foreground_mean = probs.mean(dim=(1, 2, 3))
        uncertainty_mean = uncertainty.mean(dim=(1, 2, 3))
        uncertainty_max = uncertainty.amax(dim=(1, 2, 3))
        logit_energy = coarse_lowres_logits.abs().mean(dim=(1, 2, 3))
        return torch.stack([foreground_mean, uncertainty_mean, uncertainty_max, logit_energy], dim=-1)

    @staticmethod
    def _attention_entropy(attn: torch.Tensor) -> torch.Tensor:
        safe_attn = attn.clamp(min=1e-6)
        return -(safe_attn * safe_attn.log()).sum(dim=-1)

    def _build_memory_controls(
        self,
        token: torch.Tensor,
        memory_stats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_input = torch.cat([token, memory_stats], dim=-1)
        gate_logits = self.memory_gate(gate_input)
        context_mix = torch.sigmoid(gate_logits[:, 0:1])
        residual_gate = torch.sigmoid(gate_logits[:, 1:2])
        fast_update_gate = torch.sigmoid(gate_logits[:, 2:3])
        slow_update_gate = torch.sigmoid(gate_logits[:, 3:4])
        return context_mix, residual_gate, fast_update_gate, slow_update_gate

    def forward(self, feat: torch.Tensor, coarse_lowres_logits: torch.Tensor, use_nested: bool = True):
        device = feat.device
        dtype = feat.dtype

        if not use_nested:
            zero_scalar = torch.zeros((), device=device, dtype=dtype)
            return coarse_lowres_logits, {
                "used_nested": torch.zeros((), device=device, dtype=dtype),
                "delta_mean": zero_scalar,
                "prototype_norm": 0.5
                * (
                    self.fast_prototypes.to(device=device, dtype=dtype).norm(dim=-1).mean()
                    + self.slow_prototypes.to(device=device, dtype=dtype).norm(dim=-1).mean()
                ),
                "memory_mix": zero_scalar,
                "memory_entropy": zero_scalar,
                "residual_gate": zero_scalar,
                "fast_update_gate": zero_scalar,
                "slow_update_gate": zero_scalar,
            }, None

        query_feat = self.query_proj(feat)
        token = self._compute_token(query_feat, coarse_lowres_logits)
        memory_stats = self._memory_stats(coarse_lowres_logits).to(device=device, dtype=dtype)
        context_mix, residual_gate, fast_update_gate, slow_update_gate = self._build_memory_controls(token, memory_stats)

        fast_prototypes = F.normalize(self.fast_prototypes.to(device=device, dtype=dtype), dim=-1)
        slow_prototypes = F.normalize(self.slow_prototypes.to(device=device, dtype=dtype), dim=-1)
        fast_attn = torch.softmax(torch.matmul(token, fast_prototypes.t()) / math.sqrt(self.nested_dim), dim=-1)
        slow_attn = torch.softmax(torch.matmul(token, slow_prototypes.t()) / math.sqrt(self.nested_dim), dim=-1)
        context_fast = torch.matmul(fast_attn, fast_prototypes)
        context_slow = torch.matmul(slow_attn, slow_prototypes)
        context = context_mix * context_fast + (1.0 - context_mix) * context_slow
        context = context.unsqueeze(-1).unsqueeze(-1)

        context = context.expand(-1, -1, feat.shape[-2], feat.shape[-1])
        uncertainty = self._uncertainty(coarse_lowres_logits)
        delta = self.residual_head(torch.cat([feat, context, uncertainty], dim=1))

        refined = coarse_lowres_logits + self.residual_scale * residual_gate.unsqueeze(-1) * delta
        fast_entropy = self._attention_entropy(fast_attn)
        slow_entropy = self._attention_entropy(slow_attn)
        memory_entropy = 0.5 * (fast_entropy + slow_entropy)

        nested_info = {
            "used_nested": torch.ones((), device=device, dtype=dtype),
            "delta_mean": delta.abs().mean(),
            "prototype_norm": 0.5 * (self.fast_prototypes.to(device=device, dtype=dtype).norm(dim=-1).mean() + self.slow_prototypes.to(device=device, dtype=dtype).norm(dim=-1).mean()),
            "memory_mix": context_mix.mean(),
            "memory_entropy": memory_entropy.mean(),
            "residual_gate": residual_gate.mean(),
            "fast_update_gate": fast_update_gate.mean(),
            "slow_update_gate": slow_update_gate.mean(),
        }
        nested_cache = {
            "token": token.detach(),
            "fast_attn": fast_attn.detach(),
            "slow_attn": slow_attn.detach(),
            "fast_update_gate": fast_update_gate.detach(),
            "slow_update_gate": slow_update_gate.detach(),
        }
        return refined, nested_info, nested_cache

    @torch.no_grad()
    def update_prototypes(self, nested_cache: Optional[Dict[str, torch.Tensor]], momentum: float = 0.03, max_norm: Optional[float] = None):
        if nested_cache is None:
            return
        max_norm = self.prototype_max_norm if max_norm is None else float(max_norm)
        tokens = nested_cache["token"].to(device=self.fast_prototypes.device, dtype=self.fast_prototypes.dtype)
        fast_assignments = nested_cache["fast_attn"].argmax(dim=-1)
        slow_assignments = nested_cache["slow_attn"].argmax(dim=-1)
        fast_gates = nested_cache["fast_update_gate"].to(device=self.fast_prototypes.device, dtype=self.fast_prototypes.dtype).flatten()
        slow_gates = nested_cache["slow_update_gate"].to(device=self.slow_prototypes.device, dtype=self.slow_prototypes.dtype).flatten()

        self._update_memory_bank(
            bank=self.fast_prototypes,
            tokens=tokens,
            assignments=fast_assignments,
            update_gates=fast_gates,
            momentum=float(momentum),
            max_norm=max_norm,
        )
        self._update_memory_bank(
            bank=self.slow_prototypes,
            tokens=tokens,
            assignments=slow_assignments,
            update_gates=slow_gates,
            momentum=float(momentum) * self.slow_momentum_scale,
            max_norm=max_norm,
        )

    @staticmethod
    @torch.no_grad()
    def _update_memory_bank(
        bank: torch.Tensor,
        tokens: torch.Tensor,
        assignments: torch.Tensor,
        update_gates: torch.Tensor,
        momentum: float,
        max_norm: float,
    ):
        for prototype_index in range(bank.shape[0]):
            mask = assignments == prototype_index
            if not torch.any(mask):
                continue
            weights = update_gates[mask]
            if torch.sum(weights) <= 1e-6:
                continue
            weighted_tokens = tokens[mask] * weights.unsqueeze(-1)
            target = F.normalize(weighted_tokens.sum(dim=0) / weights.sum().clamp(min=1e-6), dim=0)
            bank[prototype_index].mul_(1.0 - float(momentum)).add_(float(momentum) * target)
        norms = bank.norm(dim=-1, keepdim=True)
        scale = torch.clamp(max_norm / (norms + 1e-6), max=1.0)
        bank.mul_(scale)


class StrongBaselinePolypModel(nn.Module):
    """Clean baseline:
    - pure PyTorch ConvNeXt-like encoder
    - FPN decoder
    - aux head
    - no memory / no curriculum / no unstable branches
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        decoder_channels: int = 128,
        dropout: float = 0.1,
        encoder_name: str = "tiny_convnext",
        use_pretrained: bool = False,
        strict_pretrained: bool = False,
        pretrained_cache_dir: Optional[str] = None,
        enable_nested: bool = False,
        nested_dim: int = 128,
        nested_prototypes: int = 8,
        nested_residual_scale: float = 0.05,
        nested_max_norm: float = 1.0,
        nested_memory_hidden: int = 128,
        nested_slow_momentum_scale: float = 0.25,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.use_pretrained = bool(use_pretrained)
        self.strict_pretrained = bool(strict_pretrained)
        self.enable_nested = bool(enable_nested)
        self.encoder = build_encoder(
            encoder_name=encoder_name,
            in_channels=in_channels,
            use_pretrained=use_pretrained,
            strict_pretrained=strict_pretrained,
            pretrained_cache_dir=pretrained_cache_dir,
        )
        self.pretrained_loaded = bool(getattr(self.encoder, "pretrained_loaded", False))
        self.decoder = FPNDecoder(self.encoder.out_channels, pyramid_channels=256, seg_channels=decoder_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.seg_head = nn.Sequential(ConvBNAct(decoder_channels, decoder_channels), nn.Conv2d(decoder_channels, out_channels, 1))
        self.aux_head = nn.Sequential(ConvBNAct(decoder_channels, decoder_channels // 2), nn.Conv2d(decoder_channels // 2, out_channels, 1))
        self.nested_refiner = SafeNestedResidualRefiner(
            feat_channels=decoder_channels,
            nested_dim=nested_dim,
            num_prototypes=nested_prototypes,
            residual_scale=nested_residual_scale,
            prototype_max_norm=nested_max_norm,
            memory_hidden_dim=nested_memory_hidden,
            slow_momentum_scale=nested_slow_momentum_scale,
        )

    def get_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """
        Expose stable optimizer groups without forcing callers to depend on
        legacy attribute names like `stem` / `layer1` / ... that this model
        does not define at the top level.
        """
        encoder_params = list(self.encoder.parameters())
        encoder_param_ids = {id(param) for param in encoder_params}
        decoder_params = [param for param in self.parameters() if id(param) not in encoder_param_ids]
        return {
            "encoder": encoder_params,
            "decoder": decoder_params,
        }

    def forward(self, x: torch.Tensor, use_nested: bool = False) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        c2, c3, c4, c5 = self.encoder(x)
        fused, aux_feat = self.decoder(c2, c3, c4, c5)
        fused = self.dropout(fused)
        coarse_lowres_logits = self.seg_head(fused)
        refined_lowres_logits, nested_info, nested_cache = self.nested_refiner(
            fused,
            coarse_lowres_logits,
            use_nested=bool(use_nested and self.enable_nested),
        )
        aux_logits = self.aux_head(aux_feat)
        coarse_logits = F.interpolate(coarse_lowres_logits, size=input_size, mode="bilinear", align_corners=False)
        logits = F.interpolate(refined_lowres_logits, size=input_size, mode="bilinear", align_corners=False)
        aux_logits = F.interpolate(aux_logits, size=input_size, mode="bilinear", align_corners=False)
        return {
            "coarse_logits": coarse_logits,
            "logits": logits,
            "aux_logits": aux_logits,
            "nested_info": nested_info,
            "nested_cache": nested_cache,
        }

    @torch.no_grad()
    def update_nested_prototypes(self, nested_cache: Optional[Dict[str, torch.Tensor]], momentum: float = 0.03, max_norm: Optional[float] = None):
        self.nested_refiner.update_prototypes(nested_cache, momentum=momentum, max_norm=max_norm)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x)["logits"])

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (self.predict_proba(x) > threshold).float()
