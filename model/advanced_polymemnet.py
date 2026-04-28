"""
Advanced PolyMemnet v2.0
=======================

Integrates all 6 enhancements:
1. Cross-Stage Modulation (multi-scale interaction)
2. Adaptive CMS Scheduling (dynamic inner loop)
3. Meta Inner Optimizer (learned optimization)
4. Hierarchical Prototype Bank (tree-structured memory)
5. MC Dropout Uncertainty (better calibration)
6. Enhanced Losses (contrastive, consistency, sparsity, quality)

This is a drop-in replacement for StrongBaselinePolypModel with extensive
new capabilities controlled by flags.

Usage:
    model = AdvancedPolyMemnet(
        encoder_name="convnext_tiny",
        enable_nested=True,
        use_cross_stage_encoder=True,
        use_adaptive_cms=True,
        use_advanced_decoder=True,
        use_hierarchical_prototypes=True,
        use_enhanced_loss=True,
    )
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.strong_baseline import (
    build_encoder,
    ConvBNAct,
    FPNDecoder,
    SafeNestedResidualRefiner,
    LayerNorm2d,
)
from model.backbones.cms_decoder import CMSDecoder
from model.backbones.self_modifying_encoder_cms import CMSSelfModifyingEncoder
from model.advanced_modules import (
    CrossStageModulator,
    AdaptiveCMSSelfModifyingBlock,
    HierarchicalPrototypeBank,
    EnhancedStrongBaselineLoss,
    MetaOptimizedSelfModifyingBlock,
)


class AdvancedPolyMemnet(nn.Module):
    """
    Enhanced PolyMemnet with all improvements.

    Args:
        in_channels: input image channels (default 3)
        out_channels: output segmentation channels (default 1)
        decoder_channels: decoder feature dimension (default 128)
        dropout: dropout rate (default 0.1)
        encoder_name: encoder backbone name
        use_pretrained: load pretrained weights
        img_size: expected image size (for some backbones)

        # Nested system
        enable_nested: bool, enable nested learning
        nested_dim: prototype space dimension
        nested_prototypes: number of prototypes (flat bank)
        nested_residual_scale: refiner residual scale
        nested_max_norm: max norm for prototypes
        nested_memory_mode: "fast_slow" or "slow_only"

        # Enhancement 1: Cross-Stage Modulation
        use_cross_stage_encoder: bool, enable cross-attention between encoder stages

        # Enhancement 2: Adaptive CMS
        use_adaptive_cms: bool, dynamic inner loop parameters
        adaptive_max_steps: max inner steps for adaptive mode

        # Enhancement 3: Meta Inner Optimizer
        use_meta_optimizer: bool, replace SGD with learned optimizer (per stage)
        meta_optimizer_hidden_dim: hidden dim for meta optimizer LSTM

        # Enhancement 4: Hierarchical Prototypes
        use_hierarchical_prototypes: bool, use tree-structured bank
        hierarchical_levels: number of hierarchy levels
        prototypes_per_level: int or list of ints

        # Enhancement 5: MC Dropout
        use_mc_dropout: bool, enable MC dropout for uncertainty
        mc_dropout_p: dropout probability for MC
        num_mc_samples: number of MC samples at inference

        # Enhancement 6: Enhanced Loss
        use_enhanced_loss: bool, use multi-component loss
        prototype_contrastive_weight: weight for contrastive loss
        inner_consistency_weight: weight for inner loop consistency
        gate_sparsity_weight: weight for gate sparsity regularization
        memory_quality_weight: weight for memory quality regularization

        # Decoder CMS levels
        cms_levels: which FPN levels get CMS memory (default all)
        cross_attn_heads: number of heads for cross-stage attention
    """

    def __init__(
        self,
        # Architecture
        in_channels: int = 3,
        out_channels: int = 1,
        decoder_channels: int = 128,
        dropout: float = 0.1,
        encoder_name: str = "tiny_convnext",
        use_pretrained: bool = False,
        strict_pretrained: bool = False,
        pretrained_cache_dir: Optional[str] = None,
        img_size: Optional[int] = None,
        # Nested
        enable_nested: bool = False,
        nested_dim: int = 128,
        nested_prototypes: int = 8,
        nested_residual_scale: float = 0.15,
        nested_max_norm: float = 1.0,
        nested_memory_mode: str = "fast_slow",
        nested_memory_hidden: int = 128,
        nested_slow_momentum_scale: float = 0.25,
        # Enhancement 1: Cross-Stage
        use_cross_stage_encoder: bool = False,
        cross_attn_heads: int = 4,
        # Enhancement 2: Adaptive CMS
        use_adaptive_cms: bool = False,
        adaptive_max_steps: int = 8,
        # Enhancement 3: Meta Optimizer (applied to adaptive blocks)
        use_meta_optimizer: bool = False,
        meta_optimizer_hidden_dim: int = 128,
        # Enhancement 4: Hierarchical Prototypes
        use_hierarchical_prototypes: bool = False,
        hierarchical_levels: int = 3,
        prototypes_per_level: Union[int, List[int]] = 8,
        # Enhancement 5: MC Dropout
        use_mc_dropout: bool = False,
        mc_dropout_p: float = 0.1,
        num_mc_samples: int = 10,
        # Enhancement 6: Loss
        use_enhanced_loss: bool = False,
        prototype_contrastive_weight: float = 0.05,
        inner_consistency_weight: float = 0.02,
        gate_sparsity_weight: float = 0.01,
        memory_quality_weight: float = 0.02,
        # Decoder
        cms_levels: List[int] = [0, 1, 2, 3],
    ):
        super().__init__()

        # Store config
        self.encoder_name = encoder_name
        self.use_pretrained = use_pretrained
        self.strict_pretrained = strict_pretrained
        self.enable_nested = enable_nested
        self.use_cross_stage_encoder = use_cross_stage_encoder
        self.use_adaptive_cms = use_adaptive_cms
        self.use_meta_optimizer = use_meta_optimizer
        self.use_hierarchical_prototypes = use_hierarchical_prototypes
        self.use_mc_dropout = use_mc_dropout
        self.use_enhanced_loss = use_enhanced_loss
        self.use_advanced_decoder = enable_nested and (use_hierarchical_prototypes or use_cross_stage_encoder)

        # Build encoder backbone
        backbone = build_encoder(
            encoder_name=encoder_name,
            in_channels=in_channels,
            use_pretrained=use_pretrained,
            strict_pretrained=strict_pretrained,
            pretrained_cache_dir=pretrained_cache_dir,
            img_size=img_size,
        )
        self.pretrained_loaded = bool(getattr(backbone, "pretrained_loaded", False))
        feature_channels = backbone.out_channels if hasattr(backbone, 'out_channels') else [64, 128, 320, 512]

        # Wrap encoder with self-modifying if requested
        if enable_nested and (use_cross_stage_encoder or use_adaptive_cms or use_meta_optimizer):
            # Build stage configs
            stage_configs = []
            for idx in range(4):
                mode = "none"
                if idx >= 2:  # c4, c5
                    mode = "full"
                elif idx >= 1 and use_adaptive_cms:  # c3 with adaptive
                    mode = "full"

                cfg = {
                    "mode": mode,
                    "inner_steps": 2 if idx in [1, 2] else 4 if idx == 3 else 0,
                    "inner_lr": 5e-3 if idx == 2 else 2e-3 if idx == 3 else 1e-2,
                    "modifier_expansion": 2 if idx == 2 else 4 if idx == 3 else 1,
                    "persist_momentum": True,
                    "surprise_type": "full" if idx >= 2 else "consistency",
                }

                if use_adaptive_cms and idx >= 1 and mode == "full":
                    cfg["adaptive"] = True
                    cfg["max_adaptive_steps"] = adaptive_max_steps

                if use_meta_optimizer and idx >= 2 and mode == "full":
                    # Replace block type via class selection outside; handled in loop below
                    cfg["use_meta_optimizer"] = True

                stage_configs.append(cfg)

            self.encoder = CMSSelfModifyingEncoder(
                backbone=backbone,
                feature_channels=feature_channels,
                stage_configs=stage_configs,
                backbone_lr_decay=0.5,
                unfreeze_schedule={"deep": 0, "mid": 8, "shallow": 16},
                use_cross_stage=use_cross_stage_encoder,
                cross_modulator=None,
            )
        else:
            self.encoder = backbone

        # Build decoder
        if self.use_advanced_decoder:
            self.decoder = CMSDecoder(
                encoder_channels=feature_channels,
                fpn_channels=decoder_channels,
                num_prototypes=nested_prototypes,
                prototype_dim=nested_dim,
                fast_momentum=0.03,
                slow_momentum=0.0075,
                num_heads=4,
                cms_levels=cms_levels,
                use_hierarchical_prototypes=use_hierarchical_prototypes,
                hierarchical_levels=hierarchical_levels,
                prototypes_per_level=prototypes_per_level,
            )
            self.nested_refiner = None
        else:
            self.decoder = FPNDecoder(feature_channels, pyramid_channels=256, seg_channels=decoder_channels)
            if enable_nested:
                self.nested_refiner = SafeNestedResidualRefiner(
                    feat_channels=decoder_channels,
                    nested_dim=nested_dim,
                    num_prototypes=nested_prototypes,
                    residual_scale=nested_residual_scale,
                    prototype_max_norm=nested_max_norm,
                    memory_mode=nested_memory_mode,
                    memory_hidden_dim=nested_memory_hidden,
                    slow_momentum_scale=nested_slow_momentum_scale,
                )
            else:
                self.nested_refiner = None

        self.dropout = nn.Dropout2d(dropout)
        self.seg_head = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels),
            nn.Conv2d(decoder_channels, out_channels, 1)
        )
        self.aux_head = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels // 2),
            nn.Conv2d(decoder_channels // 2, out_channels, 1)
        )

        # MC Dropout wrapper (applied to seg_head or specific layers)
        if use_mc_dropout:
            self.mc_dropout = nn.Dropout2d(mc_dropout_p)
        else:
            self.mc_dropout = None

        # Enhanced loss
        if use_enhanced_loss:
            self.enhanced_loss_fn = EnhancedStrongBaselineLoss(
                prototype_contrastive_weight=prototype_contrastive_weight,
                inner_consistency_weight=inner_consistency_weight,
                gate_sparsity_weight=gate_sparsity_weight,
                memory_quality_weight=memory_quality_weight,
            )
        else:
            self.enhanced_loss_fn = None

    def get_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        encoder_params = list(self.encoder.parameters())
        encoder_param_ids = {id(p) for p in encoder_params}
        decoder_params = [p for p in self.parameters() if id(p) not in encoder_param_ids]
        return {"encoder": encoder_params, "decoder": decoder_params}

    def forward(
        self,
        x: torch.Tensor,
        use_nested: bool = False,
        return_uncertainty: bool = False,
        num_mc_samples: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: input image [B, 3, H, W]
            use_nested: whether to activate nested modules (if enabled)
            return_uncertainty: return uncertainty maps (requires MC dropout)
            num_mc_samples: number of MC samples (0 = use single pass)

        Returns:
            dict with logits and optional info
        """
        input_size = x.shape[-2:]
        B = x.shape[0]

        # Encoder
        encoder_out = self.encoder(x)
        if isinstance(encoder_out, (tuple, list)) and len(encoder_out) == 2:
            features, _ = encoder_out
        else:
            # Backbone returns only features
            features = encoder_out
        c2, c3, c4, c5 = features
        c2, c3, c4, c5 = features

        # Decoder
        if self.use_advanced_decoder:
            fused, aux_feat, decoder_info = self.decoder([c2, c3, c4, c5])
            nested_info = decoder_info
            nested_cache = decoder_info.get("proto_cache")
        else:
            fused, aux_feat = self.decoder(c2, c3, c4, c5)
            fused = self.dropout(fused)
            coarse_lowres_logits = self.seg_head(fused)
            if self.nested_refiner is not None and use_nested and self.enable_nested:
                refined_lowres, nested_info, nested_cache = self.nested_refiner(
                    fused, coarse_lowres_logits, use_nested=True
                )
                coarse_lowres_logits = refined_lowres
            nested_info = {}
            nested_cache = None

        # Apply MC dropout if requested
        if self.mc_dropout is not None and (self.training or num_mc_samples > 0):
            fused = self.mc_dropout(fused)

        # Segmentation heads
        coarse_logits = self.seg_head(fused)
        aux_logits = self.aux_head(aux_feat)

        # Upsample to input size
        coarse_logits = F.interpolate(coarse_logits, size=input_size, mode="bilinear", align_corners=False)
        logits = coarse_logits  # in advanced mode, no separate refinement
        aux_logits = F.interpolate(aux_logits, size=input_size, mode="bilinear", align_corners=False)

        output = {
            "coarse_logits": coarse_logits,
            "logits": logits,
            "aux_logits": aux_logits,
            "nested_info": nested_info,
            "nested_cache": nested_cache,
        }

        # MC uncertainty estimation
        if return_uncertainty and self.mc_dropout is not None and not self.training:
            with torch.no_grad():
                samples = []
                for _ in range(num_mc_samples or 10):
                    fused_aug = self.decoder([c2, c3, c4, c5])[0]
                    if self.mc_dropout:
                        fused_aug = self.mc_dropout(fused_aug)
                    sample_logits = self.seg_head(fused_aug)
                    samples.append(F.interpolate(sample_logits, size=input_size, mode="bilinear", align_corners=False))
                stacked = torch.stack(samples, dim=0)
                probs = torch.sigmoid(stacked)
                uncertainty = probs.var(dim=0)  # [B, 1, H, W]
                output["uncertainty"] = uncertainty

        return output

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        prototype_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with optional enhanced components.
        """
        if self.enhanced_loss_fn is not None:
            total_loss, loss_dict = self.enhanced_loss_fn(
                outputs=outputs,
                targets=targets,
                nested_info=outputs.get("nested_info"),
                prototype_cache=outputs.get("nested_cache"),
                prototype_labels=prototype_labels,
            )
        else:
            # Use base StrongBaselineLoss
            from loss.strong_baseline_loss import StrongBaselineLoss
            base_loss = StrongBaselineLoss()
            total_loss = base_loss(outputs, targets)
            loss_dict = {"loss_total": total_loss.item()}

        return total_loss, loss_dict

    @torch.no_grad()
    def update_nested_prototypes(self, nested_cache: Optional[Dict] = None, momentum: float = 0.03, max_norm: Optional[float] = None):
        """Update prototype banks.
        Called from training loop after each batch.
        """
        if self.use_advanced_decoder and nested_cache is not None:
            if hasattr(self.decoder, 'update_prototypes'):
                # CMSDecoder.update_prototypes expects decoder_info with 'proto_cache'
                self.decoder.update_prototypes({"proto_cache": nested_cache})
        elif self.nested_refiner is not None and nested_cache is not None:
            self.nested_refiner.update_prototypes(nested_cache, momentum=momentum, max_norm=max_norm)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, num_mc_samples: int = 0) -> torch.Tensor:
        self.eval()
        out = self.forward(x, return_uncertainty=False, num_mc_samples=num_mc_samples)
        return torch.sigmoid(out["logits"])

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5, num_mc_samples: int = 0) -> torch.Tensor:
        return (self.predict_proba(x, num_mc_samples=num_mc_samples) > threshold).float()
