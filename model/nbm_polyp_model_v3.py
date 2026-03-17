from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

STAGE_KEYS = ("s2", "s3", "s4")


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3),
            ConvBNReLU(out_channels, out_channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rates=(1, 6, 12, 18)):
        super().__init__()
        branches = []
        for rate in rates:
            if rate == 1:
                branches.append(ConvBNReLU(in_channels, out_channels, kernel_size=1))
            else:
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    )
                )
        self.branches = nn.ModuleList(branches)
        self.project = DoubleConv(out_channels * len(rates), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        return self.project(torch.cat(feats, dim=1))


class PrototypeContextModulatorSafe(nn.Module):
    def __init__(self, feat_dim: int, memory_dim: int, num_prototypes: int, hidden_dim: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to_query = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.to_fast_attn = nn.Linear(hidden_dim, num_prototypes)
        self.to_slow_attn = nn.Linear(hidden_dim, num_prototypes)
        self.context_mlp = nn.Sequential(
            nn.Linear(memory_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.to_gamma = nn.Linear(hidden_dim, feat_dim)
        self.to_beta = nn.Linear(hidden_dim, feat_dim)
        self.refine = DoubleConv(feat_dim, feat_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.to_gamma.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.normal_(self.to_beta.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, feat: torch.Tensor, fast_memory: torch.Tensor, slow_memory: torch.Tensor):
        pooled = self.pool(feat).flatten(1)
        query = self.to_query(pooled)
        fast_attn = torch.softmax(self.to_fast_attn(query), dim=1)
        slow_attn = torch.softmax(self.to_slow_attn(query), dim=1)

        fast_ctx = torch.sum(fast_attn.unsqueeze(-1) * fast_memory, dim=1)
        slow_ctx = torch.sum(slow_attn.unsqueeze(-1) * slow_memory, dim=1)
        context = torch.cat([fast_ctx, slow_ctx, fast_ctx - slow_ctx, fast_ctx * slow_ctx], dim=1)
        h = self.context_mlp(context)
        gamma = self.to_gamma(h).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(h).unsqueeze(-1).unsqueeze(-1)
        modulated = feat * (1.0 + gamma) + beta
        modulated = self.refine(modulated)
        return modulated, fast_attn, slow_attn, fast_ctx, slow_ctx


class PrototypeMemoryUpdaterSafe(nn.Module):
    def __init__(self, feat_dim: int, memory_dim: int, num_prototypes: int, hidden_dim: int = 128, write_scale: float = 0.10, erase_scale: float = 0.05, proto_max_norm: float = 0.35):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.write_scale = write_scale
        self.erase_scale = erase_scale
        self.proto_max_norm = proto_max_norm
        in_dim = feat_dim + memory_dim * 2 + 2
        self.delta_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, memory_dim),
            nn.Tanh(),
        )
        self.write_gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.erase_gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, feat, fast_memory, slow_memory, fast_attn, slow_attn, uncertainty, boundary_disagreement):
        pooled = self.pool(feat).flatten(1)
        fast_ctx = torch.sum(fast_attn.unsqueeze(-1) * fast_memory, dim=1)
        slow_ctx = torch.sum(slow_attn.unsqueeze(-1) * slow_memory, dim=1)
        if uncertainty.ndim == 0:
            uncertainty = uncertainty.view(1, 1).repeat(fast_memory.size(0), 1)
        elif uncertainty.ndim == 1:
            uncertainty = uncertainty.view(-1, 1)
        if boundary_disagreement.ndim == 0:
            boundary_disagreement = boundary_disagreement.view(1, 1).repeat(fast_memory.size(0), 1)
        elif boundary_disagreement.ndim == 1:
            boundary_disagreement = boundary_disagreement.view(-1, 1)
        z = torch.cat([pooled, fast_ctx, slow_ctx, uncertainty, boundary_disagreement], dim=1)
        delta = self.delta_mlp(z)
        write = self.write_gate(z).unsqueeze(-1)
        erase = self.erase_gate(z).unsqueeze(-1)

        delta_proto = self.write_scale * write * fast_attn.unsqueeze(-1) * delta.unsqueeze(1)
        retain = 1.0 - self.erase_scale * erase * fast_attn.unsqueeze(-1)
        updated = fast_memory * retain + delta_proto
        proto_norm = updated.norm(p=2, dim=2, keepdim=True)
        scale = torch.clamp(self.proto_max_norm / (proto_norm + 1e-6), max=1.0)
        return updated * scale


class MemoryDecoderBlockSafe(nn.Module):
    def __init__(self, in_channels, skip_channels, edge_channels, out_channels, memory_dim, num_prototypes, hidden_dim=128, max_gate: float = 0.15):
        super().__init__()
        self.reduce = ConvBNReLU(in_channels, out_channels, kernel_size=1)
        self.fuse = DoubleConv(out_channels + skip_channels, out_channels)
        self.edge_gate = nn.Sequential(
            nn.Conv2d(edge_channels, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.modulator = PrototypeContextModulatorSafe(out_channels, memory_dim, num_prototypes, hidden_dim)
        self.max_gate = max_gate
        self.gate_logit = nn.Parameter(torch.tensor(-3.0))
        self.post = DoubleConv(out_channels, out_channels)

    def memory_gate(self) -> torch.Tensor:
        return self.max_gate * torch.sigmoid(self.gate_logit)

    def forward(self, x, skip, edge, fast_memory, slow_memory):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = self.fuse(torch.cat([skip, x], dim=1))
        x = x * (1.0 + self.edge_gate(edge))
        memory_modulated, fast_attn, slow_attn, fast_ctx, slow_ctx = self.modulator(x, fast_memory, slow_memory)
        gate = self.memory_gate().view(1, 1, 1, 1)
        x = x + gate * (memory_modulated - x)
        x = self.post(x)
        return x, {"fast_attn": fast_attn, "slow_attn": slow_attn, "fast_ctx": fast_ctx, "slow_ctx": slow_ctx, "memory_gate": gate.detach()}


class EdgeAwareDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, edge_channels: int, out_channels: int):
        super().__init__()
        self.reduce = ConvBNReLU(in_channels, out_channels, kernel_size=1)
        self.fuse = DoubleConv(out_channels + skip_channels, out_channels)
        self.edge_gate = nn.Sequential(
            nn.Conv2d(edge_channels, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = DoubleConv(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = self.fuse(torch.cat([skip, x], dim=1))
        x = x * (1.0 + self.edge_gate(edge))
        return self.refine(x)


class ReverseAttentionHead(nn.Module):
    def __init__(self, feat_channels: int, mid_channels: int = 32, out_channels: int = 1, max_gate: float = 0.35):
        super().__init__()
        self.refine = nn.Sequential(
            ConvBNReLU(feat_channels + 1, feat_channels, 3),
            ConvBNReLU(feat_channels, mid_channels, 3),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
        )
        self.max_gate = max_gate
        self.gate_logit = nn.Parameter(torch.tensor(-1.5))

    def refine_gate(self):
        return self.max_gate * torch.sigmoid(self.gate_logit)

    def forward(self, feat: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        rev = 1.0 - torch.sigmoid(coarse_logits)
        rev = F.interpolate(rev, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        residual = self.refine(torch.cat([feat, rev], dim=1))
        return self.refine_gate().view(1, 1, 1, 1) * residual


class NBMPolypNetV3(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        memory_dim: int = 64,
        num_prototypes: int = 4,
        updater_hidden_dim: int = 128,
        fast_init_std: float = 2e-2,
        slow_init_std: float = 5e-3,
        decoder_memory_max_gate: float = 0.15,
        refine_max_gate: float = 0.35,
    ):
        super().__init__()
        c = base_channels
        self.memory_dim = memory_dim
        self.num_prototypes = num_prototypes
        self.fast_init_std = fast_init_std
        self.slow_init_std = slow_init_std
        self.stage_channels = {"s2": c * 2, "s3": c * 4, "s4": c * 8}

        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)
        self.aspp = ASPP(c * 16, c * 16)

        self.edge_stem = DoubleConv(in_channels, c)
        self.edge_adapt2 = ConvBNReLU(c * 2, c * 2, kernel_size=1)
        self.edge_adapt3 = ConvBNReLU(c * 4, c * 4, kernel_size=1)
        self.edge_adapt4 = ConvBNReLU(c * 8, c * 8, kernel_size=1)
        self.edge_merge2 = DoubleConv(c + c * 2, c * 2)
        self.edge_merge3 = DoubleConv(c * 2 + c * 4, c * 4)
        self.edge_merge4 = DoubleConv(c * 4 + c * 8, c * 8)

        self.dec4 = MemoryDecoderBlockSafe(c * 16, c * 8, c * 8, c * 8, memory_dim, num_prototypes, updater_hidden_dim, max_gate=decoder_memory_max_gate)
        self.dec3 = MemoryDecoderBlockSafe(c * 8, c * 4, c * 4, c * 4, memory_dim, num_prototypes, updater_hidden_dim, max_gate=decoder_memory_max_gate)
        self.dec2 = MemoryDecoderBlockSafe(c * 4, c * 2, c * 2, c * 2, memory_dim, num_prototypes, updater_hidden_dim, max_gate=decoder_memory_max_gate)
        self.dec1 = EdgeAwareDecoderBlock(c * 2, c, c, c)

        self.coarse_head = nn.Conv2d(c, out_channels, kernel_size=1)
        self.refine_head = ReverseAttentionHead(c, mid_channels=max(8, c // 2), out_channels=out_channels, max_gate=refine_max_gate)
        self.aux_head = nn.Conv2d(c * 4, out_channels, kernel_size=1)

        self.edge_proj2 = ConvBNReLU(c * 2, c, kernel_size=1)
        self.edge_proj3 = ConvBNReLU(c * 4, c, kernel_size=1)
        self.edge_proj4 = ConvBNReLU(c * 8, c, kernel_size=1)
        self.edge_head = nn.Sequential(DoubleConv(c * 4, c), nn.Conv2d(c, out_channels, kernel_size=1))

        self.updaters = nn.ModuleDict({stage: PrototypeMemoryUpdaterSafe(channels, memory_dim, num_prototypes, updater_hidden_dim) for stage, channels in self.stage_channels.items()})
        for stage in STAGE_KEYS:
            self.register_buffer(f"slow_memory_{stage}", torch.zeros(num_prototypes, memory_dim))
        self.reset_slow_memory()

    @torch.no_grad()
    def reset_slow_memory(self):
        for stage in STAGE_KEYS:
            getattr(self, f"slow_memory_{stage}").normal_(mean=0.0, std=self.slow_init_std)

    def get_slow_memory(self, stage: str, batch_size: Optional[int] = None, device=None, dtype=torch.float32):
        slow = getattr(self, f"slow_memory_{stage}")
        if device is not None or dtype is not None:
            slow = slow.to(device=device, dtype=dtype)
        if batch_size is None:
            return slow
        return slow.unsqueeze(0).repeat(batch_size, 1, 1)

    def init_memory(self, batch_size: int, device, dtype=torch.float32, from_slow: bool = True, noise_std: float = 5e-4, slow_scale: float = 0.10):
        memory = {}
        for stage in STAGE_KEYS:
            if from_slow:
                stage_memory = slow_scale * self.get_slow_memory(stage, batch_size, device, dtype)
                if noise_std > 0:
                    stage_memory = stage_memory + noise_std * torch.randn_like(stage_memory)
            else:
                stage_memory = torch.randn(batch_size, self.num_prototypes, self.memory_dim, device=device, dtype=dtype) * self.fast_init_std
            memory[stage] = stage_memory
        return memory

    @torch.no_grad()
    def summarize_memory(self, memory: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {stage: memory[stage].mean(dim=0).detach() for stage in STAGE_KEYS}

    @torch.no_grad()
    def update_slow_memory(self, task_memory_summary: Dict[str, torch.Tensor], momentum: float | Dict[str, float] = 0.02, max_norm: float = 0.60):
        for stage in STAGE_KEYS:
            stage_summary = task_memory_summary[stage]
            if stage_summary.ndim == 3:
                stage_summary = stage_summary.mean(dim=0)
            stage_summary = stage_summary.detach().to(getattr(self, f"slow_memory_{stage}").device)
            m = momentum[stage] if isinstance(momentum, dict) else momentum
            buffer = getattr(self, f"slow_memory_{stage}")
            buffer.mul_(1.0 - m).add_(m * stage_summary)
            proto_norm = buffer.norm(p=2, dim=1, keepdim=True)
            scale = torch.clamp(max_norm / (proto_norm + 1e-6), max=1.0)
            buffer.mul_(scale)

    def _compute_soft_boundary(self, prob: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(prob, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-prob, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp(0.0, 1.0)

    def _compute_update_signals(self, logits: torch.Tensor, edge_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        prob = torch.sigmoid(logits)
        edge_prob = torch.sigmoid(edge_logits)
        entropy = -(prob * torch.log(prob + 1e-7) + (1.0 - prob) * torch.log(1.0 - prob + 1e-7))
        uncertainty = entropy.mean(dim=(1, 2, 3), keepdim=False)
        pred_boundary = self._compute_soft_boundary(prob)
        boundary_disagreement = (pred_boundary - edge_prob).abs().mean(dim=(1, 2, 3), keepdim=False)
        return {"uncertainty": uncertainty, "boundary_disagreement": boundary_disagreement}

    def compute_updated_memory(self, memory_features: Dict[str, torch.Tensor], memory: Dict[str, torch.Tensor], update_signals: Dict[str, torch.Tensor], attention_cache: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        new_memory = {}
        for stage in STAGE_KEYS:
            slow_memory = self.get_slow_memory(stage, batch_size=memory[stage].size(0), device=memory[stage].device, dtype=memory[stage].dtype)
            cache = attention_cache[stage]
            new_memory[stage] = self.updaters[stage](
                feat=memory_features[stage],
                fast_memory=memory[stage],
                slow_memory=slow_memory,
                fast_attn=cache["fast_attn"],
                slow_attn=cache["slow_attn"],
                uncertainty=update_signals["uncertainty"],
                boundary_disagreement=update_signals["boundary_disagreement"],
            )
        return new_memory

    @torch.no_grad()
    def update_memory(self, memory_features, memory, update_signals, attention_cache):
        return self.compute_updated_memory(memory_features, memory, update_signals, attention_cache)

    def forward(self, x: torch.Tensor, memory: Optional[Dict[str, torch.Tensor]] = None, use_memory: bool = True) -> Dict[str, torch.Tensor]:
        if memory is None:
            memory = self.init_memory(x.size(0), x.device, x.dtype)

        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.aspp(self.down4(e4))

        edge1 = self.edge_stem(x)
        edge2 = self.edge_merge2(torch.cat([F.avg_pool2d(edge1, 2), self.edge_adapt2(e2)], dim=1))
        edge3 = self.edge_merge3(torch.cat([F.avg_pool2d(edge2, 2), self.edge_adapt3(e3)], dim=1))
        edge4 = self.edge_merge4(torch.cat([F.avg_pool2d(edge3, 2), self.edge_adapt4(e4)], dim=1))

        if use_memory:
            fast_s4 = memory["s4"]
            fast_s3 = memory["s3"]
            fast_s2 = memory["s2"]
            slow_s4 = self.get_slow_memory("s4", x.size(0), x.device, x.dtype)
            slow_s3 = self.get_slow_memory("s3", x.size(0), x.device, x.dtype)
            slow_s2 = self.get_slow_memory("s2", x.size(0), x.device, x.dtype)
        else:
            fast_s4 = torch.zeros_like(memory["s4"])
            fast_s3 = torch.zeros_like(memory["s3"])
            fast_s2 = torch.zeros_like(memory["s2"])
            slow_s4 = torch.zeros_like(fast_s4)
            slow_s3 = torch.zeros_like(fast_s3)
            slow_s2 = torch.zeros_like(fast_s2)

        d4, attn4 = self.dec4(e5, e4, edge4, fast_s4, slow_s4)
        d3, attn3 = self.dec3(d4, e3, edge3, fast_s3, slow_s3)
        d2, attn2 = self.dec2(d3, e2, edge2, fast_s2, slow_s2)
        d1 = self.dec1(d2, e1, edge1)

        coarse_logits = self.coarse_head(d1)
        refine_residual = self.refine_head(d1, coarse_logits)
        logits = coarse_logits + refine_residual
        aux_logits = F.interpolate(self.aux_head(d3), size=x.shape[-2:], mode="bilinear", align_corners=False)

        edge_fused = torch.cat([
            edge1,
            F.interpolate(self.edge_proj2(edge2), size=edge1.shape[-2:], mode="bilinear", align_corners=False),
            F.interpolate(self.edge_proj3(edge3), size=edge1.shape[-2:], mode="bilinear", align_corners=False),
            F.interpolate(self.edge_proj4(edge4), size=edge1.shape[-2:], mode="bilinear", align_corners=False),
        ], dim=1)
        edge_logits = self.edge_head(edge_fused)
        update_signals = self._compute_update_signals(logits, edge_logits)
        return {
            "logits": logits,
            "coarse_logits": coarse_logits,
            "aux_logits": aux_logits,
            "edge_logits": edge_logits,
            "memory_features": {"s4": d4, "s3": d3, "s2": d2},
            "attention_cache": {"s4": attn4, "s3": attn3, "s2": attn2} if use_memory else None,
            "update_signals": update_signals,
            "fast_memory": memory,
        }
