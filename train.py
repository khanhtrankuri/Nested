"""
Train Nested-Learning PolyMemnet v2 (Strategy B + CMS Decoder).

Kien truc moi:
  Input -> SelfModifyingEncoder(backbone + NL blocks) -> c2..c5
        -> CMSDecoder (BiFPN + CMS memory + prototype + uncertainty)
           -> fused_feat (adapted) + aux_feat (raw c5)
        -> seg_head -> logits (output CUOI, khong con coarse/refined)
        -> aux_head -> aux_logits

Inner loop (Level 2a) tu cap nhat modifier weights qua surprise objective.
CMS memory (Level 2b) cap nhat per spatial position qua linear attention.
Prototype banks (Level 2c) cap nhat qua EMA across training samples.

Usage:
  python train.py --dataset kvasir --file-path datasets/TrainDataset \
      --encoder-name pvtv2_b2 --batch-size 8 --epochs 60 --use-pretrained
"""

import argparse
import copy
import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler

from data.load_data_clean import build_clean_dataloaders, build_presplit_dataloaders
from data.load_GlaS_dataset import build_glas_dataloaders
from engine.train_eval_clean import (
    ModelEMA,
    evaluate_clean,
    test_clean,
    threshold_sweep_clean,
    train_one_epoch_clean,
)
from loss.strong_baseline_loss import StrongBaselineLoss
from loss.boundary_losses import SoftDiceBoundaryLoss, FocalBoundaryLoss, CombinedBoundaryLoss
from model.backbones.self_modifying_encoder_cms import (
    CMSSelfModifyingEncoder,
    DEFAULT_STAGE_CONFIG,
)
from model.backbones.self_modifying_encoder_minimal import MinimalCMSEncoder
from model.backbones.strong_baseline import ConvBNAct, build_encoder
from model.backbones.cms_decoder import CMSDecoder


# =============================================================================
# Model — SelfModifyingEncoder + CMSDecoder (replaces FPN + Refiner)
# =============================================================================


class NestedPolypModel(nn.Module):
    """PolyMemnet v2: Self-Modifying Encoder (Strategy B) + CMS Decoder (Option B).

    Thay doi so voi v1:
    - CMSDecoder thay the FPNDecoder + SafeNestedResidualRefiner
    - Khong con coarse_logits / refined_logits, chi co logits duy nhat
    - aux_feat = raw c5 (512-ch) thay vi smoothed p5 (128-ch)
    - Prototype EMA update thong qua CMSDecoder.update_prototypes()
    """

    def __init__(
        self,
        encoder_name: str,
        use_pretrained: bool = False,
        strict_pretrained: bool = False,
        pretrained_cache_dir: Optional[str] = None,
        img_size: Optional[int] = None,
        decoder_channels: int = 128,
        dropout: float = 0.1,
        # CMS Self-Modifying Encoder params
        encoder_variant: str = "cms",  # "cms" | "minimal"
        use_cross_stage: bool = False,
        cross_attn_heads: int = 4,
        stage_configs: Optional[List[Dict]] = None,
        backbone_lr_decay: float = 0.5,
        unfreeze_schedule: Optional[Dict[str, int]] = None,
        # CMS Decoder params
        enable_nested: bool = False,
        cms_levels: Optional[List[int]] = None,
        memory_dim: int = 64,
        num_heads: int = 4,
        gate_init_bias: float = -2.0,
        num_prototypes: int = 8,
        prototype_dim: int = 128,
        fast_momentum: float = 0.03,
        slow_momentum: float = 0.0075,
        # Multi-Scale Segmentation Heads
        enable_multi_scale: bool = False,
        multi_scale_head_mode: str = "separate",
        multi_scale_fusion: str = "weighted_sum",
        multi_scale_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.img_size = img_size
        self.enable_nested = bool(enable_nested)

        backbone = build_encoder(
            encoder_name=encoder_name,
            in_channels=3,
            use_pretrained=use_pretrained,
            strict_pretrained=strict_pretrained,
            pretrained_cache_dir=pretrained_cache_dir,
            img_size=img_size,
        )
        self.pretrained_loaded = bool(getattr(backbone, "pretrained_loaded", False))
        self.backbone_channels = list(backbone.out_channels)

        if stage_configs is None:
            stage_configs = DEFAULT_STAGE_CONFIG
        if len(stage_configs) != 4:
            raise ValueError(
                f"stage_configs must have 4 entries (c2..c5), got {len(stage_configs)}"
            )
        if cms_levels is None:
            cms_levels = [0, 1, 2, 3]
        if unfreeze_schedule is None:
            unfreeze_schedule = {"deep": 0, "mid": 8, "shallow": 16}

        self.stage_configs = list(stage_configs)
        self.unfreeze_schedule = dict(unfreeze_schedule)
        self.backbone_lr_decay = float(backbone_lr_decay)

        # Cross-stage configuration
        self.use_cross_stage = use_cross_stage
        self.cross_attn_heads = cross_attn_heads

        # Validation: cross-stage only supported for CMS encoder
        if use_cross_stage and encoder_variant != "cms":
            raise ValueError(
                f"Cross-stage modulation is only supported for encoder_variant='cms', "
                f"got '{encoder_variant}'. Set --use-cross-stage=False or use --encoder-variant cms."
            )

        # Prepare cross_modulator if needed (only for CMS encoder with cross-stage enabled)
        cross_modulator = None
        if self.use_cross_stage:
            from model.advanced_modules import CrossStageModulator
            cross_modulator = CrossStageModulator(
                channels_list=self.backbone_channels,
                num_heads=self.cross_attn_heads,
            )

        if encoder_variant == "minimal":
            self.encoder = MinimalCMSEncoder(
                backbone=backbone,
                feature_channels=self.backbone_channels,
                use_llrd=False,
                backbone_lr_decay=self.backbone_lr_decay,
                use_progressive_unfreeze=True,
                unfreeze_schedule=self.unfreeze_schedule,
                c3_adaptor_mode="light",
                use_persistent_momentum=False,
            )
        else:
            self.encoder = CMSSelfModifyingEncoder(
                backbone=backbone,
                feature_channels=self.backbone_channels,
                stage_configs=self.stage_configs,
                backbone_lr_decay=self.backbone_lr_decay,
                unfreeze_schedule=self.unfreeze_schedule,
                use_cross_stage=self.use_cross_stage,
                cross_modulator=cross_modulator,
            )

        self.decoder = CMSDecoder(
            encoder_channels=self.backbone_channels,
            fpn_channels=decoder_channels,
            num_prototypes=num_prototypes,
            prototype_dim=prototype_dim,
            fast_momentum=fast_momentum,
            slow_momentum=slow_momentum,
            num_heads=num_heads,
            cms_levels=cms_levels,
            memory_dim=memory_dim,
            gate_init_bias=gate_init_bias,
        )

        # Multi-scale segmentation heads configuration
        self.enable_multi_scale = enable_multi_scale
        self.multi_scale_head_mode = multi_scale_head_mode
        self.multi_scale_fusion = multi_scale_fusion
        self.multi_scale_weights = list(multi_scale_weights) if multi_scale_weights else [0.1, 0.2, 0.3, 0.4]
        assert len(self.multi_scale_weights) == 4, "multi_scale_weights must have 4 values"

        # Initialize multi-scale heads if enabled
        self.multi_scale_heads = None
        if self.enable_multi_scale:
            from model.multi_scale_heads import MultiScaleSegHeads
            self.multi_scale_heads = MultiScaleSegHeads(
                in_channels=decoder_channels,
                num_scales=4,
                mode=self.multi_scale_head_mode,
                fusion_type=self.multi_scale_fusion,
            )

        self.dropout = nn.Dropout2d(dropout)
        self.seg_head = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels),
            nn.Conv2d(decoder_channels, 1, 1),
        )
        # Auxiliary segmentation head from s5-level features (after BiFPN smoothing)
        # aux_feat has decoder_channels (not raw c5)
        self.aux_head = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels // 2),
            nn.Conv2d(decoder_channels // 2, 1, 1),
        )
        # Store for debugging
        self._decoder_channels = decoder_channels

    def _build_nested_info(
        self, decoder_info: Dict, device: torch.device, dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Build engine-compatible nested_info dict from decoder_info.

        The training engine (train_eval_clean.py) reads specific keys from
        nested_info for logging meters. We map CMS decoder outputs to those keys.
        """
        zero = torch.zeros((), device=device, dtype=dtype)

        # Average CMS gate values -> residual_gate meter
        cms_gates = decoder_info.get("cms_gate_values", {})
        if cms_gates:
            avg_gate = torch.stack(list(cms_gates.values())).mean()
        else:
            avg_gate = zero

        # Boundary refinement metrics
        boundary_info = decoder_info.get("boundary", {})
        boundary_gate = boundary_info.get("gate_mean", zero)
        edge_ratio = zero
        if "edge_map" in boundary_info:
            edge_map = boundary_info["edge_map"]
            # Ratio of pixels considered as edges (thresholded)
            edge_ratio = (edge_map > 0.1).float().mean()

        # Prototype norm from banks (if available)
        proto_norm = zero
        proto_ready = zero
        memory_entropy = zero
        if self.decoder.prototype is not None:
            bank = self.decoder.prototype
            fast_norm = bank.fast_prototypes.to(device=device, dtype=dtype).norm(dim=-1).mean()
            slow_norm = bank.slow_prototypes.to(device=device, dtype=dtype).norm(dim=-1).mean()
            proto_norm = 0.5 * (fast_norm + slow_norm)
            fast_ready = (bank.fast_counts > 1e-6).float().mean()
            slow_ready = (bank.slow_counts > 1e-6).float().mean()
            proto_ready = (0.5 * (fast_ready + slow_ready)).to(device=device, dtype=dtype)

            # Compute memory entropy from prototype attention weights
            attn_fast = decoder_info.get("prototype_sim_fast")
            attn_slow = decoder_info.get("prototype_sim_slow")
            if attn_fast is not None and attn_slow is not None:
                def _entropy(attn):
                    safe = attn.clamp(min=1e-6)
                    return -(safe * safe.log()).sum(dim=-1).mean()
                memory_entropy = (0.5 * (_entropy(attn_fast) + _entropy(attn_slow))).to(
                    device=device, dtype=dtype
                )

        # Prototype mix
        proto_mix = decoder_info.get("prototype_mix", zero)
        if torch.is_tensor(proto_mix):
            proto_mix = proto_mix.to(device=device, dtype=dtype)
        else:
            proto_mix = torch.tensor(float(proto_mix), device=device, dtype=dtype)

        return {
            "used_nested": torch.ones((), device=device, dtype=dtype),
            "delta_mean": zero,               # no refiner delta in v2
            "prototype_norm": proto_norm,
            "memory_mix": proto_mix,
            "memory_entropy": memory_entropy,
            "memory_quality": torch.ones((), device=device, dtype=dtype),
            "memory_ready_ratio": proto_ready,
            "residual_gate": avg_gate,
            "boundary_gate": boundary_gate,
            "edge_ratio": edge_ratio,
        }

    def forward(self, x: torch.Tensor, use_nested: bool = False) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        features, nl_info = self.encoder(x, return_nested_info=True)
        if len(features) != 4:
            raise RuntimeError(f"Expected 4 feature maps, got {len(features)}")

        # Decoder returns a dict with fused, aux_feat, smoothed_features, decoder_info
        decoder_out = self.decoder(features)
        fused = decoder_out['fused']
        aux_feat = decoder_out['aux_feat']
        decoder_info = decoder_out['decoder_info']
        smoothed_features = decoder_out.get('smoothed_features')  # List of 4 tensors or None

        # Apply dropout to fused features (used by seg_head)
        fused = self.dropout(fused)

        # Multi-scale heads OR single head
        if self.enable_multi_scale and smoothed_features is not None:
            # Use multi-scale segmentation heads
            main_logits, scale_logits, scale_logits_upsampled = self.multi_scale_heads(
                smoothed_features,
                target_size=input_size
            )
            logits = main_logits
        else:
            # Standard single segmentation head on fused features
            logits = self.seg_head(fused)
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
            scale_logits = None
            scale_logits_upsampled = None

        # Auxiliary head from c5-level feature (aux_feat)
        if aux_feat is not None:
            aux_logits = self.aux_head(aux_feat)
            aux_logits = F.interpolate(aux_logits, size=input_size, mode="bilinear", align_corners=False)
        else:
            aux_logits = None

        nested_active = bool(use_nested and self.enable_nested)
        nested_info = self._build_nested_info(decoder_info, logits.device, logits.dtype)
        if not nested_active:
            zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
            nested_info = {k: zero for k in nested_info}

        proto_cache = decoder_info.get("proto_cache")

        outputs = {
            "logits": logits,
            "coarse_logits": logits,
            "aux_logits": aux_logits,
            "nested_info": nested_info,
            "nested_cache": proto_cache,
            "nl_info": nl_info,
            "decoder_info": decoder_info,
        }

        # Add multi-scale outputs if enabled
        if self.enable_multi_scale and scale_logits is not None:
            outputs['multi_scale_logits'] = scale_logits
            outputs['multi_scale_logits_upsampled'] = scale_logits_upsampled

        return outputs

    @torch.no_grad()
    def update_nested_prototypes(
        self,
        nested_cache,
        momentum: float = 0.03,
        max_norm: Optional[float] = None,
    ):
        """Bridge for engine compatibility.

        The engine calls: model.update_nested_prototypes(outputs["nested_cache"], ...)
        We forward to CMSDecoder.update_prototypes() which uses the bank's
        configured momentum values (fast_momentum / slow_momentum).
        """
        if nested_cache is None or self.decoder.prototype is None:
            return
        self.decoder.prototype.update_prototypes(nested_cache)


# =============================================================================
# Loss wrapper — NL surprise + prototype diversity regularization
# =============================================================================


class NLLossWrapper(nn.Module):
    """StrongBaselineLoss + auxiliary surprise loss (Level 2a) +
    prototype diversity loss (Level 2c) + boundary loss.

    Surprise drives the encoder's inner-loop reconstructor.
    Prototype diversity prevents K prototypes from collapsing.
    Boundary loss improves edge accuracy (critical for small polyps).
    """

    def __init__(
        self,
        base_loss: nn.Module,
        surprise_weight: float = 0.05,
        diversity_weight: float = 0.01,
        boundary_weight: float = 0.1,
        boundary_loss_type: str = "soft_dice",
        multi_scale_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.base = base_loss
        self.surprise_weight = float(surprise_weight)
        self.diversity_weight = float(diversity_weight)
        self.boundary_weight = float(boundary_weight)
        self.multi_scale_weights = list(multi_scale_weights) if multi_scale_weights is not None else [0.1, 0.2, 0.3, 0.4]

        # Boundary loss module
        if boundary_weight > 0:
            if boundary_loss_type == "soft_dice":
                self.boundary_loss = SoftDiceBoundaryLoss(boundary_weight=2.0)
            elif boundary_loss_type == "focal":
                self.boundary_loss = FocalBoundaryLoss(boundary_weight=3.0)
            elif boundary_loss_type == "combined":
                self.boundary_loss = CombinedBoundaryLoss(
                    dice_weight=1.0,
                    focal_weight=0.5,
                    hausdorff_weight=0.1,
                )
            else:
                raise ValueError(f"Unknown boundary_loss_type: {boundary_loss_type}")
        else:
            self.boundary_loss = None

    def _prototype_diversity_loss(self, outputs: Dict) -> torch.Tensor:
        """Penalize prototype collapse: encourage pairwise dissimilarity."""
        decoder_info = outputs.get("decoder_info")
        if decoder_info is None:
            return torch.zeros((), device=outputs["logits"].device,
                               dtype=outputs["logits"].dtype)

        proto_cache = decoder_info.get("proto_cache")
        if proto_cache is None:
            return torch.zeros((), device=outputs["logits"].device,
                               dtype=outputs["logits"].dtype)

        # Compute on fast prototypes similarity matrix
        sim_fast = proto_cache.get("attn_fast")
        sim_slow = proto_cache.get("attn_slow")
        if sim_fast is None or sim_slow is None:
            return torch.zeros((), device=outputs["logits"].device,
                               dtype=outputs["logits"].dtype)

        # Diversity = encourage uniform attention (high entropy)
        # Low entropy = collapsed to single prototype
        def _neg_entropy(attn):
            # attn: (B, K) — softmax attention over prototypes
            safe = attn.clamp(min=1e-6)
            ent = -(safe * safe.log()).sum(dim=-1).mean()
            # Negate: higher entropy = lower loss
            return -ent

        loss = 0.5 * (_neg_entropy(sim_fast) + _neg_entropy(sim_slow))
        return loss

    def forward(self, outputs, targets: torch.Tensor, return_components: bool = False):
        if return_components:
            total, components = self.base(outputs, targets, return_components=True)
        else:
            total = self.base(outputs, targets, return_components=False)
            components = {}

        # --- Surprise loss (NL inner-loop, from encoder) ---
        nl_info = outputs.get("nl_info", {}) if isinstance(outputs, dict) else {}
        surprise_total = torch.zeros((), device=total.device, dtype=total.dtype)
        num_stages = 0
        for _, info in (nl_info or {}).items():
            aux = info.get("aux_surprise", None) if isinstance(info, dict) else None
            if aux is None:
                continue
            if torch.is_tensor(aux):
                surprise_total = surprise_total + aux.to(dtype=total.dtype)
                num_stages += 1

        if num_stages > 0:
            surprise_total = surprise_total / num_stages
            total = total + self.surprise_weight * surprise_total

        # --- Prototype diversity loss ---
        if isinstance(outputs, dict) and self.diversity_weight > 0:
            div_loss = self._prototype_diversity_loss(outputs)
            total = total + self.diversity_weight * div_loss
        else:
            div_loss = torch.zeros((), device=total.device, dtype=total.dtype)

        # --- Boundary loss ---
        if isinstance(outputs, dict) and self.boundary_loss is not None:
            logits = outputs.get("logits")
            if logits is not None:
                # Use targets from function argument
                # Note: targets shape (B, 1, H, W) should match logits
                b_loss = self.boundary_loss(logits, targets)
                total = total + self.boundary_weight * b_loss
            else:
                b_loss = torch.zeros((), device=total.device, dtype=total.dtype)
        else:
            b_loss = torch.zeros((), device=total.device, dtype=total.dtype)

        # --- Multi-Scale auxiliary losses ---
        if isinstance(outputs, dict) and "multi_scale_logits" in outputs:
            scale_logits = outputs["multi_scale_logits"]
            multi_scale_loss = torch.zeros((), device=total.device, dtype=total.dtype)
            for i, logits in enumerate(scale_logits):
                B, C, H_scale, W_scale = logits.shape
                # Downsample targets to match scale resolution
                if H_scale < targets.shape[-2]:
                    targets_scaled = F.interpolate(targets, size=(H_scale, W_scale), mode='nearest')
                else:
                    targets_scaled = targets
                # Compute BCE + Dice loss (lightweight, same as auxiliary)
                loss_bce_i = self.base.bce(logits, targets_scaled)
                loss_dice_i = self.base.dice(logits, targets_scaled)
                loss_i = 0.5 * loss_bce_i + 0.5 * loss_dice_i
                weight = self.multi_scale_weights[i]
                multi_scale_loss = multi_scale_loss + weight * loss_i
                if return_components:
                    components[f"loss_multi_scale_p{i+2}"] = float(loss_i.detach().item())
            total = total + multi_scale_loss
            if return_components:
                components["loss_multi_scale_total"] = float(multi_scale_loss.detach().item())

        if return_components:
            components["loss_surprise"] = float(surprise_total.detach().item())
            components["loss_diversity"] = float(div_loss.detach().item())
            components["loss_boundary"] = float(b_loss.detach().item())
            components["loss_total"] = float(total.detach().item())
            return total, components
        return total


# =============================================================================
# Argparse
# =============================================================================


def build_parser():
    parser = argparse.ArgumentParser(
        description="Nested-Learning PolyMemnet v2 training (Strategy B + CMS Decoder)."
    )
    # Data
    parser.add_argument("--dataset", choices=["kvasir", "glas"], default="kvasir")
    parser.add_argument("--file-path", default="")
    parser.add_argument("--save-root", default="")
    parser.add_argument(
        "--image-size", type=int, nargs=2, default=(384, 384), metavar=("H", "W")
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # Backbone
    parser.add_argument(
        "--encoder-name",
        choices=[
            "tiny_convnext",
            "convnext_tiny",
            "convnext_small",
            "convnext_base",
            "convnext_large",
            "pvtv2_b2",
            "pvtv2_b5",
            "convnextv2_tiny",
            "convnextv2_base",
            "convnextv2_large",
            "swinv2_tiny",
            "swinv2_small",
            "swinv2_base",
            "swinv2_large",
        ],
        default="pvtv2_b2",
    )
    parser.add_argument("--use-pretrained", action="store_true")
    parser.add_argument("--strict-pretrained", action="store_true")
    parser.add_argument("--pretrained-cache-dir", default="")
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--init-checkpoint", default="")

    # --- CMS Self-Modifying Encoder (Level 2a NL + CMS hierarchy) ---
    parser.add_argument(
        "--encoder-variant",
        choices=["cms", "minimal"],
        default="cms",
        help="'cms' = CMSSelfModifyingEncoder (full); 'minimal' = MinimalCMSEncoder (CMS ordering only).",
    )
    parser.add_argument(
        "--nl-stage-config-preset",
        choices=["default", "conservative", "aggressive", "custom"],
        default="default",
        help="Preset cho stage_configs. 'custom' = DEFAULT_STAGE_CONFIG + override CLI.",
    )
    parser.add_argument("--nl-c4-inner-steps", type=int, default=3)
    parser.add_argument("--nl-c5-inner-steps", type=int, default=6)
    parser.add_argument("--nl-c4-inner-lr", type=float, default=5e-4)
    parser.add_argument("--nl-c5-inner-lr", type=float, default=2e-2)
    parser.add_argument("--backbone-lr-decay", type=float, default=0.65)
    parser.add_argument("--unfreeze-deep-epoch", type=int, default=0)
    parser.add_argument("--unfreeze-mid-epoch", type=int, default=8)
    parser.add_argument("--unfreeze-shallow-epoch", type=int, default=16)
    parser.add_argument("--adaptor-lr", type=float, default=5e-4)
    parser.add_argument("--nl-surprise-weight", type=float, default=0.05)

    # --- CMS Decoder (Level 2b memory + Level 2c prototype) ---
    parser.add_argument("--enable-nested", action="store_true")
    parser.add_argument("--nested-start-epoch", type=int, default=20)
    parser.add_argument(
        "--cms-levels", type=int, nargs="+", default=[0, 1, 2, 3],
        help="FPN levels to apply CMS memory (0=p2, 1=p3, 2=p4, 3=p5)",
    )
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--gate-init-bias", type=float, default=-2.0)
    parser.add_argument("--num-prototypes", type=int, default=8)
    parser.add_argument("--prototype-dim", type=int, default=128)
    parser.add_argument("--fast-momentum", type=float, default=0.05)
    parser.add_argument("--slow-momentum", type=float, default=0.003)
    parser.add_argument("--diversity-weight", type=float, default=0.01)

    # --- Boundary Loss ---
    parser.add_argument("--boundary-weight", type=float, default=0.1,
                        help="Weight for boundary-aware loss (0 to disable)")
    parser.add_argument("--boundary-loss-type", choices=["soft_dice", "focal", "combined"],
                        default="soft_dice", help="Type of boundary loss")

    # --- Multi-Scale Segmentation Heads ---
    parser.add_argument("--multi-scale-heads", action="store_true",
                        help="Enable segmentation heads at multiple FPN levels (p2,p3,p4,p5)")
    parser.add_argument("--multi-scale-head-mode", choices=["separate", "shared"], default="separate",
                        help="Separate: distinct head per scale; Shared: same head for all")
    parser.add_argument("--multi-scale-weights", type=float, nargs=4, default=[0.1, 0.2, 0.3, 0.4],
                        help="Loss weights for p2, p3, p4, p5 auxiliary heads")
    parser.add_argument("--multi-scale-fusion", choices=["weighted_sum", "attention"], default="weighted_sum",
                        help="How to fuse multi-scale logits for final output")

    # --- Cross-Stage Modulation ---
    parser.add_argument("--use-cross-stage", action="store_true",
                        help="Enable cross-stage attention between encoder stages")
    parser.add_argument("--cross-attn-heads", type=int, default=4,
                        help="Number of attention heads for cross-stage modulation")

    # --- Engine-level nested control ---
    parser.add_argument(
        "--skip-nested-if-hurts",
        dest="skip_nested_if_hurts",
        action="store_true",
    )
    parser.add_argument(
        "--no-skip-nested-if-hurts",
        dest="skip_nested_if_hurts",
        action="store_false",
    )
    parser.set_defaults(skip_nested_if_hurts=True)
    parser.add_argument("--nested-skip-margin", type=float, default=0.002)
    parser.add_argument("--nested-momentum", type=float, default=0.05)
    parser.add_argument("--nested-max-norm", type=float, default=1.0)

    # Split + optimization
    parser.add_argument("--protocol", choices=["strict", "kfold"], default="strict")
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
    )

    # Eval / TTA / EMA
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--tta-scales", type=float, nargs="+", default=[1.0, 0.75, 1.25])
    parser.add_argument(
        "--eval-nested-mode", choices=["auto", "on", "off"], default="auto"
    )
    parser.add_argument(
        "--test-nested-mode", choices=["auto", "on", "off"], default="auto"
    )
    parser.add_argument("--patience", type=int, default=5)

    # Data sampling
    parser.add_argument("--small-polyp-sampling-power", type=float, default=0.25)
    parser.add_argument(
        "--stratified-split", dest="stratified_split", action="store_true"
    )
    parser.add_argument(
        "--no-stratified-split", dest="stratified_split", action="store_false"
    )
    parser.set_defaults(stratified_split=True)
    parser.add_argument(
        "--glas-val-ratio",
        type=float,
        default=0.0,
        help="GlaS: fraction of train to hold out as val (0 = use testA as val)",
    )

    # Smoke-test helpers
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a minimal sanity-check loop (overrides epochs/steps).",
    )
    parser.add_argument(
        "--smoke-max-steps", type=int, default=2, help="Batches per epoch in smoke-test."
    )
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="Disable mixed precision (useful for debugging NL inner loop).",
    )
    parser.set_defaults(use_amp=True)
    return parser


# =============================================================================
# Utilities
# =============================================================================


def _resolve_nested_usage(mode: str, nested_active: bool) -> bool:
    if mode == "auto":
        return bool(nested_active)
    if mode == "on":
        return True
    if mode == "off":
        return False
    raise ValueError(f"Unsupported nested usage mode: {mode}")


def _build_stage_configs(args) -> List[Dict]:
    """Map --nl-stage-config-preset (+ overrides) to a per-stage config list."""
    if args.nl_stage_config_preset == "default":
        return copy.deepcopy(DEFAULT_STAGE_CONFIG)

    if args.nl_stage_config_preset == "conservative":
        return [
            {"mode": "none"},
            {"mode": "none"},
            {
                "mode": "light",
                "lora_rank": 8,
                "residual_init": 0.05,
            },
            {
                "mode": "full",
                "inner_steps": 2,
                "inner_lr": 1e-2,
                "inner_momentum": 0.9,
                "modifier_expansion": 2,
                "surprise_type": "full",
                "persist_momentum": True,
                "residual_init": 0.05,
            },
        ]

    if args.nl_stage_config_preset == "aggressive":
        return [
            {"mode": "light", "lora_rank": 4, "residual_init": 0.03},
            {
                "mode": "full",
                "inner_steps": 1,
                "inner_lr": 2e-3,
                "inner_momentum": 0.85,
                "modifier_expansion": 1,
                "surprise_type": "consistency",
                "persist_momentum": False,
                "residual_init": 0.03,
            },
            {
                "mode": "full",
                "inner_steps": 3,
                "inner_lr": 1e-2,
                "inner_momentum": 0.9,
                "modifier_expansion": 2,
                "surprise_type": "full",
                "persist_momentum": True,
                "residual_init": 0.05,
            },
            {
                "mode": "full",
                "inner_steps": 5,
                "inner_lr": 3e-2,
                "inner_momentum": 0.95,
                "modifier_expansion": 4,
                "surprise_type": "full",
                "persist_momentum": True,
                "residual_init": 0.08,
            },
        ]

    cfg = copy.deepcopy(DEFAULT_STAGE_CONFIG)
    cfg[2]["inner_steps"] = int(args.nl_c4_inner_steps)
    cfg[2]["inner_lr"] = float(args.nl_c4_inner_lr)
    cfg[3]["inner_steps"] = int(args.nl_c5_inner_steps)
    cfg[3]["inner_lr"] = float(args.nl_c5_inner_lr)
    return cfg


def _limit_loader_steps(loader, max_steps: int):
    """Yield only the first ``max_steps`` batches of ``loader``."""

    class _LimitedLoader:
        def __init__(self, inner, limit):
            self._inner = inner
            self._limit = limit

        def __iter__(self):
            for idx, batch in enumerate(self._inner):
                if idx >= self._limit:
                    break
                yield batch

        def __len__(self):
            return min(len(self._inner), self._limit)

    return _LimitedLoader(loader, max_steps)


# =============================================================================
# Main
# =============================================================================


def main():
    args = build_parser().parse_args()

    # --- Dataset-specific defaults ---
    if args.dataset == "glas":
        if not args.file_path:
            args.file_path = "datasets/Warwick_QU_Dataset"
        if not args.save_root:
            args.save_root = "outputs/glas_nested_learning_v2"
    else:
        if not args.file_path:
            args.file_path = "datasets/Adenocarcinoma"
        if not args.save_root:
            args.save_root = "outputs/kvasir_nested_learning_v2"

    if args.smoke_test:
        args.epochs = max(1, min(args.epochs, 1))
        args.warmup_epochs = 0
        args.patience = args.epochs
        args.num_workers = 0

    os.makedirs(args.save_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Build dataloaders ---
    test_loaders = {}
    if args.dataset == "glas":
        (
            train_loader,
            val_loader,
            testA_loader,
            testB_loader,
            meta_info,
        ) = build_glas_dataloaders(
            root_dir=args.file_path,
            image_size=tuple(args.image_size),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            train_augmentation=True,
            val_ratio=args.glas_val_ratio,
        )
        test_loaders["testA"] = testA_loader
        test_loaders["testB"] = testB_loader
    else:
        train_loader, val_loader, test_loader, meta_info = build_clean_dataloaders(
            file_path=args.file_path,
            image_size=tuple(args.image_size),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            protocol=args.protocol,
            fold_index=args.fold_index,
            num_folds=args.num_folds,
            train_augmentation=True,
            stratified_split=args.stratified_split,
            small_polyp_sampling_power=args.small_polyp_sampling_power,
        )
        test_loaders["test"] = test_loader
        # train_loader, val_loader, test_loader, meta_info = build_presplit_dataloaders(
        #     root=args.file_path,
        #     image_size=tuple(args.image_size),
        #     batch_size=args.batch_size,
        #     num_workers=args.num_workers,
        #     train_augmentation=True,
        #     small_polyp_sampling_power=args.small_polyp_sampling_power,
        # )
        # test_loaders["test"] = test_loader

    if args.smoke_test:
        train_loader = _limit_loader_steps(train_loader, args.smoke_max_steps)
        val_loader = _limit_loader_steps(val_loader, args.smoke_max_steps)
        test_loaders = {
            name: _limit_loader_steps(loader, args.smoke_max_steps)
            for name, loader in test_loaders.items()
        }

    pretrained_cache_dir = (
        args.pretrained_cache_dir.strip()
        or os.path.join(args.save_root, ".torch_cache")
    )

    stage_configs = _build_stage_configs(args)
    unfreeze_schedule = {
        "deep": int(args.unfreeze_deep_epoch),
        "mid": int(args.unfreeze_mid_epoch),
        "shallow": int(args.unfreeze_shallow_epoch),
    }

    model_kwargs = dict(
        encoder_name=args.encoder_name,
        use_pretrained=args.use_pretrained,
        strict_pretrained=args.strict_pretrained,
        pretrained_cache_dir=pretrained_cache_dir,
        img_size=args.image_size[0],
        decoder_channels=args.decoder_channels,
        dropout=args.dropout,
        encoder_variant=args.encoder_variant,
        stage_configs=stage_configs,
        backbone_lr_decay=float(args.backbone_lr_decay),
        unfreeze_schedule=unfreeze_schedule,
        # Cross-Stage Modulation
        use_cross_stage=args.use_cross_stage,
        cross_attn_heads=args.cross_attn_heads,
        # CMS Decoder
        enable_nested=args.enable_nested,
        cms_levels=list(args.cms_levels),
        memory_dim=args.memory_dim,
        num_heads=args.num_heads,
        gate_init_bias=args.gate_init_bias,
        num_prototypes=args.num_prototypes,
        prototype_dim=args.prototype_dim,
        fast_momentum=args.fast_momentum,
        slow_momentum=args.slow_momentum,
        # Multi-Scale Segmentation Heads
        enable_multi_scale=args.multi_scale_heads,
        multi_scale_head_mode=args.multi_scale_head_mode,
        multi_scale_fusion=args.multi_scale_fusion,
        multi_scale_weights=args.multi_scale_weights,
    )
    model = NestedPolypModel(**model_kwargs).to(device)
    criterion = NLLossWrapper(
        StrongBaselineLoss(),
        surprise_weight=args.nl_surprise_weight,
        diversity_weight=args.diversity_weight,
        boundary_weight=args.boundary_weight,
        boundary_loss_type=args.boundary_loss_type,
        multi_scale_weights=args.multi_scale_weights if args.multi_scale_heads else None,
    )

    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        incompatible = model.load_state_dict(state_dict, strict=False)
        print(
            f"[NestedPolyp v2] Loaded init checkpoint: {args.init_checkpoint} | "
            f"missing={list(incompatible.missing_keys)[:10]} | "
            f"unexpected={list(incompatible.unexpected_keys)[:10]}"
        )

    meta_info["model"] = {
        **{k: v for k, v in model_kwargs.items() if k != "pretrained_cache_dir"},
        "pretrained_loaded": bool(getattr(model, "pretrained_loaded", False)),
        "architecture": "NestedPolypModel_v2_CMSDecoder",
    }
    with open(os.path.join(args.save_root, "meta_info.json"), "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    encoder_groups = model.encoder.build_param_groups(
        base_backbone_lr=args.encoder_lr,
        adaptor_lr=args.adaptor_lr,
        modifier_inner_lr_scale=0.5,
        weight_decay=args.weight_decay,
    )
    encoder_param_ids = {id(p) for g in encoder_groups for p in g["params"]}
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
    if not encoder_groups or not decoder_params:
        raise RuntimeError("Empty parameter group detected.")

    all_groups = encoder_groups + [
        {
            "params": decoder_params,
            "lr": args.decoder_lr,
            "weight_decay": args.weight_decay,
            "name": "decoder",
        }
    ]

    print(
        f"[NestedPolyp v2] encoder={args.encoder_name} | pretrained={args.use_pretrained} "
        f"| pretrained_loaded={getattr(model, 'pretrained_loaded', False)} "
        f"| preset={args.nl_stage_config_preset} | unfreeze={unfreeze_schedule} "
        f"| enable_nested={args.enable_nested}"
    )
    print(
        f"[NestedPolyp v2] CMS: levels={args.cms_levels} | mem_dim={args.memory_dim} "
        f"| heads={args.num_heads} | prototypes={args.num_prototypes} "
        f"| proto_dim={args.prototype_dim}"
    )
    encoder_total = sum(p.numel() for g in encoder_groups for p in g["params"])
    print(
        f"[NestedPolyp v2] params: encoder={encoder_total:,} | "
        f"decoder={sum(p.numel() for p in decoder_params):,} | "
        f"total={sum(p.numel() for p in model.parameters()):,}"
    )
    print("[Optimizer] Parameter groups:")
    for g in all_groups:
        n = sum(p.numel() for p in g["params"])
        print(
            f"  {g.get('name', '?'):<28} lr={g['lr']:.2e}  "
            f"wd={g['weight_decay']:.1e}  params={n:,}"
        )

    optimizer = torch.optim.AdamW(all_groups)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.25,
        total_iters=max(args.warmup_epochs, 1),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - args.warmup_epochs, 1),
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[args.warmup_epochs],
    )

    scaler = GradScaler("cuda", enabled=args.use_amp and torch.cuda.is_available())
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

    best_state = None
    best_val = {"iou": -1.0, "dice": -1.0, "threshold": 0.5}
    best_nested_active = False
    epochs_without_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.encoder.set_epoch(epoch)
        if epoch == 1 or epoch % 5 == 0:
            print(model.encoder.describe())

        nested_active = bool(
            args.enable_nested and epoch >= args.nested_start_epoch
        )
        val_nested_active = _resolve_nested_usage(args.eval_nested_mode, nested_active)
        train_metrics = train_one_epoch_clean(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            scaler=scaler,
            use_amp=args.use_amp,
            grad_clip=1.0,
            ema=ema,
            print_freq=20,
            use_nested=nested_active,
            skip_nested_if_hurts=args.skip_nested_if_hurts,
            nested_skip_margin=args.nested_skip_margin,
            nested_momentum=args.nested_momentum,
            nested_max_norm=args.nested_max_norm,
        )
        scheduler.step()

        eval_model = ema.ema if ema is not None else model
        val_best = threshold_sweep_clean(
            model=eval_model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            thresholds=args.thresholds,
            use_amp=args.use_amp,
            use_tta=args.use_tta,
            tta_scales=args.tta_scales,
            use_nested=val_nested_active,
        )

        test_metrics_epoch = {}
        if args.dataset == "glas":
            best_thr = val_best["threshold"]
            for split_name, loader in test_loaders.items():
                split_metrics = evaluate_clean(
                    model=eval_model,
                    loader=loader,
                    criterion=criterion,
                    device=device,
                    threshold=best_thr,
                    use_amp=args.use_amp,
                    use_tta=args.use_tta,
                    tta_scales=args.tta_scales,
                    use_nested=val_nested_active,
                )
                test_metrics_epoch[split_name] = split_metrics

        current_lr = optimizer.param_groups[0]["lr"]
        history_entry = {
            "epoch": epoch,
            "lr": current_lr,
            "nested_active": nested_active,
            "val_nested_active": val_nested_active,
            "train": train_metrics,
            "val": val_best,
        }
        if test_metrics_epoch:
            history_entry["test"] = test_metrics_epoch
        history.append(history_entry)

        test_log = ""
        for split_name, m in test_metrics_epoch.items():
            test_log += (
                f" | {split_name}_iou={m['iou']:.4f}"
                f" | {split_name}_dice={m['dice']:.4f}"
            )
        print(
            f"\n[Epoch {epoch}] lr={current_lr:.7f} | "
            f"train_iou={train_metrics['iou']:.4f} | "
            f"val_iou={val_best['iou']:.4f} | val_dice={val_best['dice']:.4f} | "
            f"best_thr={val_best['threshold']:.2f} | "
            f"nested_active={nested_active} | val_nested={val_nested_active}"
            f"{test_log}\n"
        )

        improved = False
        if val_best["iou"] > best_val["iou"]:
            improved = True
        elif val_best["iou"] == best_val["iou"] and val_best["dice"] > best_val["dice"]:
            improved = True

        if improved:
            best_val = val_best
            best_nested_active = val_nested_active
            best_state = copy.deepcopy(
                (ema.ema if ema is not None else model).state_dict()
            )
            torch.save(
                {
                    "state_dict": best_state,
                    "best_val": best_val,
                    "best_nested_active": best_nested_active,
                    "model_kwargs": {
                        k: v for k, v in model_kwargs.items() if k != "pretrained_cache_dir"
                    },
                    "train_config": vars(args),
                },
                os.path.join(args.save_root, "best_model.pth"),
            )
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        with open(os.path.join(args.save_root, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if epochs_without_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
            break

    if best_state is None:
        print("[NestedPolyp v2] No improvement over initial val; saving final state instead.")
        best_state = copy.deepcopy(
            (ema.ema if ema is not None else model).state_dict()
        )

    model.load_state_dict(best_state)
    test_nested_active = _resolve_nested_usage(args.test_nested_mode, best_nested_active)

    all_test_metrics = {}
    for split_name, loader in test_loaders.items():
        metrics = test_clean(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            save_dir=os.path.join(args.save_root, f"predictions_{split_name}"),
            threshold=best_val["threshold"],
            use_amp=args.use_amp,
            use_tta=args.use_tta,
            tta_scales=args.tta_scales,
            use_nested=test_nested_active,
        )
        all_test_metrics[split_name] = metrics
        print(
            f"[{split_name}] IoU: {metrics['iou']:.4f} | Dice: {metrics['dice']:.4f}"
        )

    with open(os.path.join(args.save_root, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_test_metrics, f, indent=2)

    print(f"\nBest validation IoU: {best_val['iou']:.4f}")
    print(f"Best validation Dice: {best_val['dice']:.4f}")
    print(f"Best threshold: {best_val['threshold']:.2f}")
    print(f"Best nested active: {best_nested_active}")
    print(f"Final test nested active: {test_nested_active}")


if __name__ == "__main__":
    main()
