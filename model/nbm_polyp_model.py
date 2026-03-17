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


class MemoryFiLMModulator(nn.Module):
    def __init__(self, feat_dim: int, memory_dim: int, hidden_dim: int = 128):
        super().__init__()
        context_dim = memory_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
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

    def forward(self, feat: torch.Tensor, fast_memory: torch.Tensor, slow_memory: torch.Tensor) -> torch.Tensor:
        context = torch.cat(
            [
                fast_memory,
                slow_memory,
                fast_memory - slow_memory,
                fast_memory * slow_memory,
            ],
            dim=1,
        )
        h = self.mlp(context)
        gamma = self.to_gamma(h).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(h).unsqueeze(-1).unsqueeze(-1)
        out = feat * (1.0 + gamma) + beta
        return self.refine(out)


class StageMemoryUpdater(nn.Module):
    def __init__(self, feat_dim: int, memory_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_dim = feat_dim + memory_dim + memory_dim + 2
        self.delta_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, memory_dim),
            nn.Tanh(),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, memory_dim),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        feat: torch.Tensor,
        memory: torch.Tensor,
        slow_memory: torch.Tensor,
        uncertainty: torch.Tensor,
        boundary_disagreement: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.pool(feat).flatten(1)
        if uncertainty.ndim == 0:
            uncertainty = uncertainty.view(1, 1).repeat(memory.size(0), 1)
        elif uncertainty.ndim == 1:
            uncertainty = uncertainty.view(-1, 1)
        if boundary_disagreement.ndim == 0:
            boundary_disagreement = boundary_disagreement.view(1, 1).repeat(memory.size(0), 1)
        elif boundary_disagreement.ndim == 1:
            boundary_disagreement = boundary_disagreement.view(-1, 1)

        z = torch.cat([pooled, memory, slow_memory, uncertainty, boundary_disagreement], dim=1)
        delta = self.delta_mlp(z)
        gate = self.gate_mlp(z)
        candidate = memory + delta
        return gate * memory + (1.0 - gate) * candidate


class MemoryDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        edge_channels: int,
        out_channels: int,
        memory_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.reduce = ConvBNReLU(in_channels, out_channels, kernel_size=1)
        self.fuse = DoubleConv(out_channels + skip_channels, out_channels)
        self.edge_gate = nn.Sequential(
            nn.Conv2d(edge_channels, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.memory_fusion = MemoryFiLMModulator(
            feat_dim=out_channels,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        edge: torch.Tensor,
        fast_memory: torch.Tensor,
        slow_memory: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = self.fuse(torch.cat([skip, x], dim=1))
        gate = self.edge_gate(edge)
        x = x * (1.0 + gate)
        x = self.memory_fusion(x, fast_memory, slow_memory)
        return x


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


class NBMPolypNet(nn.Module):
    """
    Nested Boundary-guided Multi-scale Memory Network for polyp segmentation.

    - U-Net-like semantic encoder/decoder
    - Dedicated edge guidance branch
    - Multi-scale fast/slow memory at decoder scales s4(1/8), s3(1/4), s2(1/2)
    - Label-free memory update using uncertainty + edge disagreement
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        memory_dim: int = 64,
        updater_hidden_dim: int = 128,
        fast_init_std: float = 5e-2,
        slow_init_std: float = 1e-2,
    ):
        super().__init__()
        c = base_channels
        self.memory_dim = memory_dim
        self.fast_init_std = fast_init_std
        self.slow_init_std = slow_init_std
        self.stage_channels = {
            "s2": c * 2,
            "s3": c * 4,
            "s4": c * 8,
        }

        # semantic encoder
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)

        # edge guidance branch
        self.edge_stem = DoubleConv(in_channels, c)
        self.edge_adapt2 = ConvBNReLU(c * 2, c * 2, kernel_size=1)
        self.edge_adapt3 = ConvBNReLU(c * 4, c * 4, kernel_size=1)
        self.edge_adapt4 = ConvBNReLU(c * 8, c * 8, kernel_size=1)
        self.edge_merge2 = DoubleConv(c + c * 2, c * 2)
        self.edge_merge3 = DoubleConv(c * 2 + c * 4, c * 4)
        self.edge_merge4 = DoubleConv(c * 4 + c * 8, c * 8)

        # decoder with memory
        self.dec4 = MemoryDecoderBlock(c * 16, c * 8, c * 8, c * 8, memory_dim, updater_hidden_dim)
        self.dec3 = MemoryDecoderBlock(c * 8, c * 4, c * 4, c * 4, memory_dim, updater_hidden_dim)
        self.dec2 = MemoryDecoderBlock(c * 4, c * 2, c * 2, c * 2, memory_dim, updater_hidden_dim)
        self.dec1 = EdgeAwareDecoderBlock(c * 2, c, c, c)

        self.out_head = nn.Conv2d(c, out_channels, kernel_size=1)
        self.aux_head = nn.Conv2d(c * 4, out_channels, kernel_size=1)

        self.edge_proj2 = ConvBNReLU(c * 2, c, kernel_size=1)
        self.edge_proj3 = ConvBNReLU(c * 4, c, kernel_size=1)
        self.edge_proj4 = ConvBNReLU(c * 8, c, kernel_size=1)
        self.edge_head = nn.Sequential(
            DoubleConv(c * 4, c),
            nn.Conv2d(c, out_channels, kernel_size=1),
        )

        self.updaters = nn.ModuleDict(
            {
                stage: StageMemoryUpdater(
                    feat_dim=channels,
                    memory_dim=memory_dim,
                    hidden_dim=updater_hidden_dim,
                )
                for stage, channels in self.stage_channels.items()
            }
        )

        for stage in STAGE_KEYS:
            self.register_buffer(f"slow_memory_{stage}", torch.zeros(memory_dim))
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
        return slow.unsqueeze(0).repeat(batch_size, 1)

    def init_memory(
        self,
        batch_size: int,
        device,
        dtype=torch.float32,
        from_slow: bool = True,
        noise_std: float = 1e-3,
        slow_scale: float = 0.2,
    ) -> Dict[str, torch.Tensor]:
        memory = {}
        for stage in STAGE_KEYS:
            if from_slow:
                stage_memory = slow_scale * self.get_slow_memory(stage, batch_size, device, dtype)
                if noise_std > 0:
                    stage_memory = stage_memory + noise_std * torch.randn_like(stage_memory)
            else:
                stage_memory = torch.randn(batch_size, self.memory_dim, device=device, dtype=dtype) * self.fast_init_std
            memory[stage] = stage_memory
        return memory

    @torch.no_grad()
    def summarize_memory(self, memory: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {stage: memory[stage].mean(dim=0).detach() for stage in STAGE_KEYS}

    @torch.no_grad()
    def update_slow_memory(
        self,
        task_memory_summary: Dict[str, torch.Tensor],
        momentum: float | Dict[str, float] = 0.05,
        max_norm: float = 1.0,
    ):
        for stage in STAGE_KEYS:
            stage_summary = task_memory_summary[stage]
            if stage_summary.ndim == 2:
                stage_summary = stage_summary.mean(dim=0)
            stage_summary = stage_summary.detach().to(getattr(self, f"slow_memory_{stage}").device)
            m = momentum[stage] if isinstance(momentum, dict) else momentum
            buffer = getattr(self, f"slow_memory_{stage}")
            buffer.mul_(1.0 - m).add_(m * stage_summary)
            norm = buffer.norm(p=2)
            if norm > max_norm:
                buffer.mul_(max_norm / (norm + 1e-6))

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
        return {
            "uncertainty": uncertainty,
            "boundary_disagreement": boundary_disagreement,
        }

    def compute_updated_memory(
        self,
        memory_features: Dict[str, torch.Tensor],
        memory: Dict[str, torch.Tensor],
        update_signals: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        new_memory = {}
        for stage in STAGE_KEYS:
            slow_memory = self.get_slow_memory(
                stage,
                batch_size=memory[stage].size(0),
                device=memory[stage].device,
                dtype=memory[stage].dtype,
            )
            new_memory[stage] = self.updaters[stage](
                feat=memory_features[stage],
                memory=memory[stage],
                slow_memory=slow_memory,
                uncertainty=update_signals["uncertainty"],
                boundary_disagreement=update_signals["boundary_disagreement"],
            )
        return new_memory

    @torch.no_grad()
    def update_memory(
        self,
        memory_features: Dict[str, torch.Tensor],
        memory: Dict[str, torch.Tensor],
        update_signals: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self.compute_updated_memory(memory_features, memory, update_signals)

    def forward(self, x: torch.Tensor, memory: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # semantic encoder
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        # edge branch
        edge1 = self.edge_stem(x)
        edge2 = self.edge_merge2(torch.cat([F.avg_pool2d(edge1, 2), self.edge_adapt2(e2)], dim=1))
        edge3 = self.edge_merge3(torch.cat([F.avg_pool2d(edge2, 2), self.edge_adapt3(e3)], dim=1))
        edge4 = self.edge_merge4(torch.cat([F.avg_pool2d(edge3, 2), self.edge_adapt4(e4)], dim=1))

        slow_s4 = self.get_slow_memory("s4", x.size(0), x.device, x.dtype)
        slow_s3 = self.get_slow_memory("s3", x.size(0), x.device, x.dtype)
        slow_s2 = self.get_slow_memory("s2", x.size(0), x.device, x.dtype)

        d4 = self.dec4(e5, e4, edge4, memory["s4"], slow_s4)
        d3 = self.dec3(d4, e3, edge3, memory["s3"], slow_s3)
        d2 = self.dec2(d3, e2, edge2, memory["s2"], slow_s2)
        d1 = self.dec1(d2, e1, edge1)

        logits = self.out_head(d1)
        aux_logits = F.interpolate(self.aux_head(d3), size=x.shape[-2:], mode="bilinear", align_corners=False)

        edge_fused = torch.cat(
            [
                edge1,
                F.interpolate(self.edge_proj2(edge2), size=edge1.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(self.edge_proj3(edge3), size=edge1.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(self.edge_proj4(edge4), size=edge1.shape[-2:], mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        edge_logits = self.edge_head(edge_fused)

        update_signals = self._compute_update_signals(logits, edge_logits)

        return {
            "logits": logits,
            "aux_logits": aux_logits,
            "edge_logits": edge_logits,
            "memory_features": {
                "s4": d4,
                "s3": d3,
                "s2": d2,
            },
            "update_signals": update_signals,
            "slow_memory": {
                "s4": slow_s4,
                "s3": slow_s3,
                "s2": slow_s2,
            },
            "fast_memory": memory,
        }

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, memory: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        logits = self.forward(x, memory)["logits"]
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, memory: Dict[str, torch.Tensor], threshold: float = 0.5) -> torch.Tensor:
        return (self.predict_proba(x, memory) > threshold).float()
