from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        activation: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = ConvBNAct(out_channels, out_channels, kernel_size=3, activation=False)
        if in_channels != out_channels or stride != 1:
            self.shortcut = ConvBNAct(in_channels, out_channels, kernel_size=1, stride=stride, activation=False)
        else:
            self.shortcut = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return self.act(out)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = ConvBNAct(channels, channels, kernel_size=3, groups=channels)
        self.pointwise = ConvBNAct(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class SpatialReductionAttention2d(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, sr_ratio: int = 1, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.sr_ratio = sr_ratio

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, bias=False)
        else:
            self.sr = None
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _reshape_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = tensor.shape
        tensor = tensor.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def _scaled_dot_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return torch.matmul(attn, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        q = self._reshape_tokens(self.q(self.norm_q(tokens)))

        if self.sr is not None:
            kv_source = self.sr(x).flatten(2).transpose(1, 2)
        else:
            kv_source = tokens
        kv = self.kv(self.norm_kv(kv_source))
        kv = kv.view(batch_size, kv_source.size(1), 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = self._scaled_dot_attn(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch_size, -1, channels)
        attn = self.dropout(self.proj(attn))
        return attn.transpose(1, 2).reshape(batch_size, channels, height, width)


class HybridTransformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, sr_ratio: int = 1, mlp_ratio: float = 2.0):
        super().__init__()
        hidden_dim = int(channels * mlp_ratio)
        self.local = DepthwiseSeparableBlock(channels)
        self.attn = SpatialReductionAttention2d(channels, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )
        self.gamma_attn = nn.Parameter(torch.full((channels,), 1e-2))
        self.gamma_mlp = nn.Parameter(torch.full((channels,), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.local(x)
        x = x + self.gamma_attn.view(1, -1, 1, 1) * self.attn(x)

        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.gamma_mlp.view(1, 1, -1) * self.mlp(self.norm(tokens))
        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)


class DetailStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        hidden = max(out_channels // 2, 16)
        self.block1 = ResidualConvBlock(in_channels, hidden)
        self.block2 = ResidualConvBlock(hidden, hidden)
        self.block3 = ResidualConvBlock(hidden, out_channels)
        self.down_half = ResidualConvBlock(out_channels, out_channels, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.block1(x)
        x = self.block2(x)
        full = self.block3(x)
        return full, self.down_half(full)


class HybridHierarchicalEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.stage1 = nn.Sequential(
            ConvBNAct(in_channels, c, kernel_size=3, stride=2),
            ResidualConvBlock(c, c),
            ResidualConvBlock(c, c),
        )
        self.stage2 = nn.Sequential(
            ResidualConvBlock(c, c * 2, stride=2),
            ResidualConvBlock(c * 2, c * 2),
        )
        self.stage3_down = ResidualConvBlock(c * 2, c * 4, stride=2)
        self.stage3_blocks = nn.Sequential(
            HybridTransformerBlock(c * 4, num_heads=4, sr_ratio=2),
            ResidualConvBlock(c * 4, c * 4),
        )
        self.stage4_down = ResidualConvBlock(c * 4, c * 8, stride=2)
        self.stage4_blocks = nn.Sequential(
            HybridTransformerBlock(c * 8, num_heads=8, sr_ratio=1),
            ResidualConvBlock(c * 8, c * 8),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.stage1(x)
        e2 = self.stage2(e1)
        e3 = self.stage3_blocks(self.stage3_down(e2))
        e4 = self.stage4_blocks(self.stage4_down(e3))
        return e1, e2, e3, e4


class PyramidPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_scales: Iterable[int] = (1, 2, 4)):
        super().__init__()
        inter_channels = max(out_channels // max(len(tuple(pool_scales)), 1), 8)
        self.pool_scales = tuple(pool_scales)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    ConvBNAct(in_channels, inter_channels, kernel_size=1),
                )
                for scale in self.pool_scales
            ]
        )
        self.project = ResidualConvBlock(in_channels + inter_channels * len(self.pool_scales), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for stage in self.stages:
            pooled = stage(x)
            pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
            features.append(pooled)
        return self.project(torch.cat(features, dim=1))


class ASPPLite(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilations: Iterable[int] = (1, 3, 5)):
        super().__init__()
        dilations = tuple(dilations)
        inter_channels = max(out_channels // max(len(dilations), 1), 8)
        branches = []
        for dilation in dilations:
            if dilation == 1:
                branches.append(ConvBNAct(in_channels, inter_channels, kernel_size=1))
            else:
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            inter_channels,
                            kernel_size=3,
                            padding=dilation,
                            dilation=dilation,
                            bias=False,
                        ),
                        nn.BatchNorm2d(inter_channels),
                        nn.GELU(),
                    )
                )
        self.branches = nn.ModuleList(branches)
        self.project = ResidualConvBlock(inter_channels * len(dilations), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        return self.project(torch.cat(features, dim=1))


class HybridContextNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.ppm = PyramidPooling(in_channels, out_channels)
        self.aspp = ASPPLite(in_channels, out_channels)
        self.project = ResidualConvBlock(in_channels + out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ppm = self.ppm(x)
        aspp = self.aspp(x)
        return self.project(torch.cat([x, ppm, aspp], dim=1))


class PartialNestedSkipBridge(nn.Module):
    def __init__(self, skip_channels: int, support_channels: int, out_channels: int):
        super().__init__()
        self.skip_proj = ConvBNAct(skip_channels, out_channels, kernel_size=1)
        self.support_proj = ConvBNAct(support_channels, out_channels, kernel_size=1)
        self.refine = ResidualConvBlock(out_channels * 2, out_channels)

    def forward(self, skip: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        support = F.interpolate(support, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat([self.skip_proj(skip), self.support_proj(support)], dim=1))


class EdgeGuidanceBranch(nn.Module):
    def __init__(self, detail_channels: int, e1_channels: int, e2_channels: int, guide_channels: int):
        super().__init__()
        self.detail_proj = ConvBNAct(detail_channels, guide_channels, kernel_size=3)
        self.e1_proj = ConvBNAct(e1_channels, guide_channels, kernel_size=1)
        self.e2_proj = ConvBNAct(e2_channels, guide_channels, kernel_size=1)
        self.fuse = ResidualConvBlock(guide_channels * 3, guide_channels)
        self.down_half = ResidualConvBlock(guide_channels, guide_channels, stride=2)
        self.down_quarter = ResidualConvBlock(guide_channels, guide_channels, stride=2)
        self.down_eighth = ResidualConvBlock(guide_channels, guide_channels, stride=2)
        self.edge_head = nn.Sequential(
            ResidualConvBlock(guide_channels, guide_channels),
            nn.Conv2d(guide_channels, 1, kernel_size=1),
        )

    def forward(
        self,
        detail_full: torch.Tensor,
        e1: torch.Tensor,
        e2: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        e1_up = F.interpolate(self.e1_proj(e1), size=detail_full.shape[-2:], mode="bilinear", align_corners=False)
        e2_up = F.interpolate(self.e2_proj(e2), size=detail_full.shape[-2:], mode="bilinear", align_corners=False)
        full = self.fuse(torch.cat([self.detail_proj(detail_full), e1_up, e2_up], dim=1))
        half = self.down_half(full)
        quarter = self.down_quarter(half)
        eighth = self.down_eighth(quarter)
        edge_logits = self.edge_head(full)
        return {"full": full, "half": half, "quarter": quarter, "eighth": eighth}, edge_logits


class EdgeAwareDecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        edge_channels: int,
        nested_channels: int = 0,
    ):
        super().__init__()
        self.in_proj = ConvBNAct(in_channels, out_channels, kernel_size=1)
        self.skip_proj = ConvBNAct(skip_channels, out_channels, kernel_size=1)
        self.nested_proj = ConvBNAct(nested_channels, out_channels, kernel_size=1) if nested_channels > 0 else None
        branch_count = 2 + int(self.nested_proj is not None)
        self.fuse = ResidualConvBlock(out_channels * branch_count, out_channels)
        self.edge_feat = ConvBNAct(edge_channels, out_channels, kernel_size=1)
        self.edge_gate = nn.Sequential(nn.Conv2d(edge_channels, out_channels, kernel_size=1), nn.Sigmoid())
        self.refine = ResidualConvBlock(out_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        edge: torch.Tensor,
        nested: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        parts = [self.in_proj(x), self.skip_proj(skip)]
        if self.nested_proj is not None:
            if nested is None:
                raise ValueError("nested input is required for this decoder stage.")
            nested = F.interpolate(nested, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            parts.append(self.nested_proj(nested))

        fused = self.fuse(torch.cat(parts, dim=1))
        edge = F.interpolate(edge, size=fused.shape[-2:], mode="bilinear", align_corners=False)
        fused = fused + self.edge_feat(edge) * self.edge_gate(edge)
        return self.refine(fused)


class FinalRefinementStage(nn.Module):
    def __init__(self, in_channels: int, detail_channels: int, edge_channels: int, out_channels: int):
        super().__init__()
        self.x_proj = ConvBNAct(in_channels, out_channels, kernel_size=1)
        self.detail_proj = ConvBNAct(detail_channels, out_channels, kernel_size=1)
        self.edge_proj = ConvBNAct(edge_channels, out_channels, kernel_size=1)
        self.edge_gate = nn.Sequential(nn.Conv2d(edge_channels, out_channels, kernel_size=1), nn.Sigmoid())
        self.fuse = ResidualConvBlock(out_channels * 3, out_channels)
        self.refine = ResidualConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, detail: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        edge = F.interpolate(edge, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([self.x_proj(x), self.detail_proj(detail), self.edge_proj(edge)], dim=1))
        fused = fused * (1.0 + self.edge_gate(edge))
        return self.refine(fused)


class ReverseAttentionRefiner(nn.Module):
    def __init__(self, feat_channels: int, mid_channels: int, out_channels: int = 1, max_residual: float = 0.35):
        super().__init__()
        self.pre = ResidualConvBlock(feat_channels + 2, feat_channels)
        self.head = nn.Sequential(
            ConvBNAct(feat_channels, mid_channels, kernel_size=3),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
        )
        self.max_residual = max_residual
        self.residual_logit = nn.Parameter(torch.tensor(-1.5))

    def forward(self, feat: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        coarse_prob = torch.sigmoid(coarse_logits)
        reverse = 1.0 - coarse_prob
        uncertainty = 1.0 - torch.abs(2.0 * coarse_prob - 1.0)
        stacked = torch.cat([feat, reverse, uncertainty], dim=1)
        residual = self.head(self.pre(stacked))
        scale = self.max_residual * torch.sigmoid(self.residual_logit)
        return scale.view(1, 1, 1, 1) * residual


class SafePrototypeMemoryUnit(nn.Module):
    def __init__(
        self,
        feat_channels: int,
        memory_dim: int,
        num_prototypes: int,
        hidden_dim: int = 128,
        residual_scale: float = 0.10,
        update_rate: float = 0.08,
        proto_max_norm: float = 1.0,
        slow_pull: float = 0.02,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_prototypes = num_prototypes
        self.residual_scale = residual_scale
        self.update_rate = update_rate
        self.proto_max_norm = proto_max_norm
        self.slow_pull = slow_pull

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.query_proj = nn.Sequential(
            nn.Linear(feat_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, memory_dim),
        )
        self.context_fuse = nn.Sequential(
            nn.Linear(memory_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, memory_dim),
        )
        self.delta_proj = nn.Linear(memory_dim, feat_channels)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.update_candidate = nn.Sequential(
            nn.Linear(memory_dim * 3 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, memory_dim),
        )
        self.update_gate = nn.Sequential(
            nn.Linear(memory_dim * 3 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _attend(self, query: torch.Tensor, prototypes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query = F.normalize(query, dim=-1)
        prototypes_norm = F.normalize(prototypes, dim=-1)
        logits = torch.einsum("bd,bpd->bp", query, prototypes_norm)
        attn = logits.softmax(dim=1)
        context = torch.einsum("bp,bpd->bd", attn, prototypes)
        return attn, context

    def forward(
        self,
        feat: torch.Tensor,
        fast_memory: torch.Tensor,
        slow_memory: torch.Tensor,
        use_memory: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        query = self.query_proj(self.pool(feat).flatten(1))
        fast_attn, fast_context = self._attend(query, fast_memory)
        slow_attn, slow_context = self._attend(query, slow_memory)

        fused_context = self.context_fuse(
            torch.cat([query, fast_context, slow_context, slow_context - fast_context], dim=1)
        )
        delta = torch.tanh(self.delta_proj(fused_context)).unsqueeze(-1).unsqueeze(-1)
        residual = self.residual_scale * self.spatial_gate(feat) * delta
        if not use_memory:
            residual = torch.zeros_like(residual)
        refined = feat + residual

        info = {
            "query": query,
            "fast_attn": fast_attn,
            "slow_attn": slow_attn,
            "fast_context": fast_context,
            "slow_context": slow_context,
            "residual_mean": residual.abs().mean(dim=(1, 2, 3)),
        }
        return refined, info

    @torch.no_grad()
    def update_memory(
        self,
        fast_memory: torch.Tensor,
        slow_memory: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        uncertainty: torch.Tensor,
        boundary_disagreement: torch.Tensor,
    ) -> torch.Tensor:
        if uncertainty.ndim == 1:
            uncertainty = uncertainty.unsqueeze(1)
        if boundary_disagreement.ndim == 1:
            boundary_disagreement = boundary_disagreement.unsqueeze(1)

        query = cache["query"]
        fast_context = cache["fast_context"]
        slow_context = cache["slow_context"]
        fast_attn = cache["fast_attn"]

        update_input = torch.cat([query, fast_context, slow_context, uncertainty, boundary_disagreement], dim=1)
        candidate = self.update_candidate(update_input)
        gate = self.update_gate(update_input)

        confidence = (1.0 - uncertainty).clamp(0.0, 1.0) * (1.0 - boundary_disagreement).clamp(0.0, 1.0)
        step = self.update_rate * gate * confidence
        updated = fast_memory + step.unsqueeze(-1) * fast_attn.unsqueeze(-1) * (candidate.unsqueeze(1) - fast_memory)
        updated = (1.0 - self.slow_pull) * updated + self.slow_pull * slow_memory

        proto_norm = updated.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.proto_max_norm / (proto_norm + 1e-6), max=1.0)
        return updated * scale
