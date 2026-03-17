from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules_hybrid import (
    DetailStem,
    EdgeAwareDecoderStage,
    EdgeGuidanceBranch,
    FinalRefinementStage,
    HybridContextNeck,
    HybridHierarchicalEncoder,
    PartialNestedSkipBridge,
    ReverseAttentionRefiner,
    SafePrototypeMemoryUnit,
)


MEMORY_STAGE_KEYS = ("stage2", "stage3")


class NBMPolypNetHybrid(nn.Module):
    """
    Practical hybrid polyp segmenter with a strong standalone backbone and
    conservative prototype memory used only as a residual correction.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        memory_dim: int = 64,
        num_prototypes: int = 6,
        memory_blend: float = 0.10,
        slow_memory_max_norm: float = 1.0,
        fast_init_std: float = 2e-2,
        slow_init_std: float = 5e-3,
        decoder_memory_scale: float = 0.10,
    ):
        super().__init__()
        c = base_channels
        self.out_channels = out_channels
        self.memory_dim = memory_dim
        self.num_prototypes = num_prototypes
        self.memory_blend = memory_blend
        self.slow_memory_max_norm = slow_memory_max_norm
        self.fast_init_std = fast_init_std
        self.slow_init_std = slow_init_std
        self.stage_channels = {"stage3": c * 4, "stage2": c * 2}

        self.detail_stem = DetailStem(in_channels, c)
        self.encoder = HybridHierarchicalEncoder(in_channels=in_channels, base_channels=base_channels)
        self.context_neck = HybridContextNeck(c * 8, c * 8)

        self.edge_branch = EdgeGuidanceBranch(detail_channels=c, e1_channels=c, e2_channels=c * 2, guide_channels=c)

        self.nested_24 = PartialNestedSkipBridge(skip_channels=c * 2, support_channels=c * 4, out_channels=c * 2)
        self.nested_12 = PartialNestedSkipBridge(skip_channels=c, support_channels=c * 2, out_channels=c)

        self.dec4 = EdgeAwareDecoderStage(
            in_channels=c * 8,
            skip_channels=c * 4,
            out_channels=c * 4,
            edge_channels=c,
        )
        self.dec3 = EdgeAwareDecoderStage(
            in_channels=c * 4,
            skip_channels=c * 2,
            out_channels=c * 2,
            edge_channels=c,
            nested_channels=c * 2,
        )
        self.dec2 = EdgeAwareDecoderStage(
            in_channels=c * 2,
            skip_channels=c,
            out_channels=c,
            edge_channels=c,
            nested_channels=c,
        )
        self.final_refine = FinalRefinementStage(in_channels=c, detail_channels=c, edge_channels=c, out_channels=c)

        self.memory_units = nn.ModuleDict(
            {
                "stage3": SafePrototypeMemoryUnit(
                    feat_channels=c * 4,
                    memory_dim=memory_dim,
                    num_prototypes=num_prototypes,
                    residual_scale=decoder_memory_scale,
                    proto_max_norm=max(1.0, slow_memory_max_norm * 1.25),
                ),
                "stage2": SafePrototypeMemoryUnit(
                    feat_channels=c * 2,
                    memory_dim=memory_dim,
                    num_prototypes=num_prototypes,
                    residual_scale=decoder_memory_scale,
                    proto_max_norm=max(1.0, slow_memory_max_norm * 1.25),
                ),
            }
        )

        self.coarse_head = nn.Conv2d(c, out_channels, kernel_size=1)
        self.side_head_stage3 = nn.Conv2d(c * 4, out_channels, kernel_size=1)
        self.side_head_stage2 = nn.Conv2d(c * 2, out_channels, kernel_size=1)
        self.reverse_refine = ReverseAttentionRefiner(
            feat_channels=c,
            mid_channels=max(c // 2, 16),
            out_channels=out_channels,
            max_residual=0.35,
        )

        for stage in MEMORY_STAGE_KEYS:
            self.register_buffer(f"slow_memory_{stage}", torch.zeros(num_prototypes, memory_dim))
        self.reset_slow_memory()

    @torch.no_grad()
    def reset_slow_memory(self):
        for stage in MEMORY_STAGE_KEYS:
            getattr(self, f"slow_memory_{stage}").normal_(mean=0.0, std=self.slow_init_std)

    def get_slow_memory(
        self,
        stage: str,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        memory = getattr(self, f"slow_memory_{stage}")
        if device is not None or dtype is not None:
            memory = memory.to(device=device or memory.device, dtype=dtype or memory.dtype)
        if batch_size is None:
            return memory
        return memory.unsqueeze(0).repeat(batch_size, 1, 1)

    def init_memory(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        from_slow: bool = True,
        noise_std: float = 5e-4,
    ) -> Dict[str, torch.Tensor]:
        memory: Dict[str, torch.Tensor] = {}
        for stage in MEMORY_STAGE_KEYS:
            if from_slow:
                stage_memory = self.get_slow_memory(stage, batch_size=batch_size, device=device, dtype=dtype)
                if noise_std > 0:
                    stage_memory = stage_memory + noise_std * torch.randn_like(stage_memory)
            else:
                stage_memory = torch.randn(
                    batch_size,
                    self.num_prototypes,
                    self.memory_dim,
                    device=device,
                    dtype=dtype,
                ) * self.fast_init_std
            memory[stage] = stage_memory
        return memory

    @torch.no_grad()
    def summarize_memory(self, memory: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        summary = {}
        for stage, value in memory.items():
            if value.ndim == 3:
                summary[stage] = value.mean(dim=0).detach()
            elif value.ndim == 2:
                summary[stage] = value.detach()
            else:
                raise ValueError(f"Unexpected memory shape for {stage}: {tuple(value.shape)}")
        return summary

    @torch.no_grad()
    def update_slow_memory(
        self,
        task_memory_summary: Dict[str, torch.Tensor],
        momentum: float | Dict[str, float] = 0.05,
        max_norm: Optional[float] = None,
    ):
        max_norm = self.slow_memory_max_norm if max_norm is None else max_norm
        for stage in MEMORY_STAGE_KEYS:
            summary = task_memory_summary[stage]
            if summary.ndim == 3:
                summary = summary.mean(dim=0)
            summary = summary.detach().to(getattr(self, f"slow_memory_{stage}").device)
            m = momentum[stage] if isinstance(momentum, dict) else float(momentum)
            buffer = getattr(self, f"slow_memory_{stage}")
            buffer.mul_(1.0 - m).add_(m * summary)
            norms = buffer.norm(dim=-1, keepdim=True)
            scale = torch.clamp(max_norm / (norms + 1e-6), max=1.0)
            buffer.mul_(scale)

    def _soft_boundary(self, prob: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(prob, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-prob, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp(0.0, 1.0)

    def _compute_update_signals(
        self,
        logits: torch.Tensor,
        coarse_logits: torch.Tensor,
        edge_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        final_prob = torch.sigmoid(logits)
        coarse_prob = torch.sigmoid(coarse_logits)
        edge_prob = torch.sigmoid(edge_logits)

        uncertainty = (1.0 - torch.abs(2.0 * final_prob - 1.0)).mean(dim=(1, 2, 3))
        refinement_gap = (final_prob - coarse_prob).abs().mean(dim=(1, 2, 3))
        boundary_disagreement = (self._soft_boundary(final_prob) - edge_prob).abs().mean(dim=(1, 2, 3))

        return {
            "uncertainty": uncertainty.clamp(0.0, 1.0),
            "refinement_gap": refinement_gap.clamp(0.0, 1.0),
            "boundary_disagreement": boundary_disagreement.clamp(0.0, 1.0),
        }

    @torch.no_grad()
    def compute_updated_memory(
        self,
        memory_features: Dict[str, torch.Tensor],
        memory: Dict[str, torch.Tensor],
        update_signals: Dict[str, torch.Tensor],
        attention_cache: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        updated = {}
        uncertainty = update_signals["uncertainty"]
        boundary_disagreement = update_signals["boundary_disagreement"]
        for stage in MEMORY_STAGE_KEYS:
            slow = self.get_slow_memory(
                stage,
                batch_size=memory[stage].size(0),
                device=memory[stage].device,
                dtype=memory[stage].dtype,
            )
            updated[stage] = self.memory_units[stage].update_memory(
                fast_memory=memory[stage],
                slow_memory=slow,
                cache=attention_cache[stage],
                uncertainty=uncertainty,
                boundary_disagreement=boundary_disagreement,
            )
        return updated

    @torch.no_grad()
    def update_memory(
        self,
        memory_features: Dict[str, torch.Tensor],
        memory: Dict[str, torch.Tensor],
        update_signals: Dict[str, torch.Tensor],
        attention_cache: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        return self.compute_updated_memory(
            memory_features=memory_features,
            memory=memory,
            update_signals=update_signals,
            attention_cache=attention_cache,
        )

    def _empty_memory_cache(self, feat: torch.Tensor, memory: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = feat.size(0)
        device = feat.device
        return {
            "query": torch.zeros(batch_size, self.memory_dim, device=device, dtype=feat.dtype),
            "fast_attn": torch.zeros(batch_size, self.num_prototypes, device=device, dtype=feat.dtype),
            "slow_attn": torch.zeros(batch_size, self.num_prototypes, device=device, dtype=feat.dtype),
            "fast_context": torch.zeros(batch_size, self.memory_dim, device=device, dtype=feat.dtype),
            "slow_context": torch.zeros(batch_size, self.memory_dim, device=device, dtype=feat.dtype),
            "residual_mean": torch.zeros(batch_size, device=device, dtype=feat.dtype),
        }

    def _prepare_memory(self, memory: Dict[str, torch.Tensor], batch_size: int, device, dtype) -> Dict[str, torch.Tensor]:
        prepared = {}
        for stage in MEMORY_STAGE_KEYS:
            value = memory[stage]
            if value.ndim == 3:
                if value.size(0) == batch_size:
                    prepared[stage] = value.to(device=device, dtype=dtype)
                else:
                    prepared[stage] = value.mean(dim=0, keepdim=True).to(device=device, dtype=dtype).repeat(batch_size, 1, 1)
            elif value.ndim == 2:
                prepared[stage] = value.to(device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            else:
                raise ValueError(f"Unexpected memory shape for {stage}: {tuple(value.shape)}")
        return prepared

    def _memory_norm(self, memory: Optional[torch.Tensor], slow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if memory is None:
            fast_norm = torch.zeros((), device=slow.device, dtype=slow.dtype)
        else:
            fast_norm = memory.norm(dim=-1).mean()
        slow_norm = slow.norm(dim=-1).mean()
        return fast_norm, slow_norm

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[Dict[str, torch.Tensor]] = None,
        use_memory: bool = True,
        disable_memory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        memory_enabled = bool(use_memory and not disable_memory)
        if memory is None:
            memory = self.init_memory(batch_size=x.size(0), device=x.device, dtype=x.dtype, noise_std=0.0)
        else:
            memory = self._prepare_memory(memory, batch_size=x.size(0), device=x.device, dtype=x.dtype)

        detail_full, _ = self.detail_stem(x)
        e1, e2, e3, e4 = self.encoder(x)
        bottleneck = self.context_neck(e4)
        edge_features, edge_logits = self.edge_branch(detail_full, e1, e2)

        nested_24 = self.nested_24(e2, e3)
        d4_base = self.dec4(bottleneck, e3, edge_features["eighth"])
        slow_stage3 = self.get_slow_memory("stage3", batch_size=x.size(0), device=x.device, dtype=x.dtype)
        if memory_enabled:
            d4, cache_stage3 = self.memory_units["stage3"](d4_base, memory["stage3"], slow_stage3, use_memory=True)
        else:
            d4, cache_stage3 = d4_base, self._empty_memory_cache(d4_base, memory["stage3"])

        d3_base = self.dec3(d4, e2, edge_features["quarter"], nested=nested_24)
        slow_stage2 = self.get_slow_memory("stage2", batch_size=x.size(0), device=x.device, dtype=x.dtype)
        if memory_enabled:
            d3, cache_stage2 = self.memory_units["stage2"](d3_base, memory["stage2"], slow_stage2, use_memory=True)
        else:
            d3, cache_stage2 = d3_base, self._empty_memory_cache(d3_base, memory["stage2"])

        nested_12 = self.nested_12(e1, d3)
        d2 = self.dec2(d3, e1, edge_features["half"], nested=nested_12)
        d1 = self.final_refine(d2, detail_full, edge_features["full"])

        coarse_logits = self.coarse_head(d1)
        final_logits = coarse_logits + self.reverse_refine(d1, coarse_logits)
        aux_logits = [
            F.interpolate(self.side_head_stage3(d4), size=x.shape[-2:], mode="bilinear", align_corners=False),
            F.interpolate(self.side_head_stage2(d3), size=x.shape[-2:], mode="bilinear", align_corners=False),
        ]

        update_signals = self._compute_update_signals(final_logits, coarse_logits, edge_logits)
        if memory_enabled:
            stage3_fast_norm, stage3_slow_norm = self._memory_norm(memory.get("stage3"), slow_stage3[0])
            stage2_fast_norm, stage2_slow_norm = self._memory_norm(memory.get("stage2"), slow_stage2[0])
        else:
            zero_norm = torch.zeros((), device=x.device, dtype=x.dtype)
            stage3_fast_norm, stage2_fast_norm = zero_norm, zero_norm
            stage3_slow_norm = slow_stage3[0].norm(dim=-1).mean()
            stage2_slow_norm = slow_stage2[0].norm(dim=-1).mean()
        memory_delta = 0.5 * (cache_stage3["residual_mean"].mean() + cache_stage2["residual_mean"].mean())

        return {
            "coarse_logits": coarse_logits,
            "logits": final_logits,
            "edge_logits": edge_logits,
            "aux_logits": aux_logits,
            "memory_features": {"stage3": d4, "stage2": d3},
            "attention_cache": {"stage3": cache_stage3, "stage2": cache_stage2} if memory_enabled else None,
            "update_signals": update_signals,
            "memory_state": memory,
            "memory_info": {
                "used_memory": torch.tensor(float(memory_enabled), device=x.device, dtype=x.dtype),
                "memory_delta": memory_delta,
                "fast_norm_stage3": stage3_fast_norm,
                "slow_norm_stage3": stage3_slow_norm,
                "fast_norm_stage2": stage2_fast_norm,
                "slow_norm_stage2": stage2_slow_norm,
            },
        }
