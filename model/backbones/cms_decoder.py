"""
CMS Decoder — Option B: BiFPN + Continuum Memory System + Prototype Memory
===========================================================================
Thay thế hoàn toàn SafeNestedResidualRefiner bằng cách nhúng adaptive memory,
prototype banks, và uncertainty gating trực tiếp vào BiFPN decoder.

Paper: "Nested Learning: The Illusion of Deep Learning Architectures"
       (Behrouz et al., NeurIPS 2025), §7 Continuum Memory System, §8 Hope.

NL Paradigm Mapping:
    Level 0 (f=0):       Encoder backbone weights (frozen/slow fine-tune)
    Level 1 (f=1/batch):  BiFPN weights, CMS projections, gates, seg_head, prototype projections
    Level 2a (f=1/sample): Encoder self-modification (surprise-based inner loop)
    Level 2b (f=1/token):  CMS memory states (linear attention, updated per spatial position)
    Level 2c (f=1/batch):  Prototype banks (EMA across training samples)

Knowledge Transfer:
    Encoder → Decoder:     features (direct parametric, NL §3.3)
    CMS memory ↔ Prototype: prototype context augments CMS output (direct non-parametric)
    Decoder → seg_head:    adapted features (direct parametric)
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Shared building blocks (same as strong_baseline.py to avoid circular imports)
# =============================================================================

class _ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class _BiFPNFuseCell(nn.Module):
    """BiFPN-style learnable weighted fusion of two feature maps."""
    def __init__(self, channels: int, eps: float = 1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2))
        self.eps = eps
        self.conv = _ConvBNAct(channels, channels)

    def forward(self, feat_a: Tensor, feat_b: Tensor) -> Tensor:
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        return self.conv(w[0] * feat_a + w[1] * feat_b)


# =============================================================================
# A) CMS Memory Module — Linear Attention memory per FPN level
# =============================================================================

class CMSMemoryModule(nn.Module):
    """
    Continuum Memory System for a single FPN level.

    NL mapping (§7):
        Level 1 (outer): W_k, W_v, W_q projections — trained by outer optimizer
        Level 2 (inner): Memory state M_t — updated per spatial position via
                         linear attention accumulation

    Formulation (parallel linear attention, non-causal):
        phi(·) = ELU(·) + 1  (positive feature map)
        M = M_0 + Σ_n phi(k_n) ⊗ v_n      (accumulate over spatial positions)
        Z = Z_0 + Σ_n phi(k_n)             (normalization term)
        y_n = M^T phi(q_n) / (Z^T phi(q_n))  (retrieval per position)

    M_0 is meta-learned (Level 1 parameter): provides a warm-start memory that
    captures dataset-level priors, refined by the outer optimizer.

    Different FPN levels naturally have different "update frequencies":
        p2 (1/4 res): ~9216 tokens → fast, dense spatial memory
        p3 (1/8 res): ~2304 tokens → medium
        p4 (1/16 res): ~576 tokens → slow, semantic memory
        p5 (1/32 res): ~144 tokens → slowest, global context

    Args:
        channels: number of channels at this FPN level
        memory_dim: dimension of key/value/query projections (default: 64)
        num_heads: multi-head count for memory (default: 4)
        gate_bias: initial bias for output gate (default: -2.0 → gate ≈ 0.12)
    """

    def __init__(
        self,
        channels: int,
        memory_dim: int = 64,
        num_heads: int = 4,
        gate_bias: float = -2.0,
    ):
        super().__init__()
        self.channels = channels
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        assert memory_dim % num_heads == 0, \
            f"memory_dim ({memory_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = memory_dim // num_heads

        self.norm = nn.GroupNorm(min(8, channels // 4), channels)
        self.k_proj = nn.Conv2d(channels, memory_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(channels, memory_dim, 1, bias=False)
        self.q_proj = nn.Conv2d(channels, memory_dim, 1, bias=False)
        self.out_proj = nn.Conv2d(memory_dim, channels, 1)

        # Meta-learned initial memory state M_0 (Level 1, trained by outer optimizer)
        # Per head: (H, head_dim, head_dim) — represents accumulated key-value associations
        self.M_0 = nn.Parameter(torch.zeros(num_heads, self.head_dim, self.head_dim))
        # Initial normalization term Z_0
        self.Z_0 = nn.Parameter(torch.ones(num_heads, self.head_dim) * 0.1)

        # Output gate — starts small to avoid disrupting BiFPN features
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, 1),
        )
        nn.init.constant_(self.gate_net[-1].bias, gate_bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass: linear attention memory retrieval with gated residual.

        Args:
            x: feature map (B, C, H, W) from BiFPN level

        Returns:
            output: x + gate * memory_retrieval (B, C, H, W)
            gate_value: scalar gate value for logging (B, 1, 1, 1)
        """
        B, C, H, W = x.shape
        N = H * W
        residual = x
        x_norm = self.norm(x)

        # Project to key, value, query — reshape to multi-head
        k = self.k_proj(x_norm).view(B, self.num_heads, self.head_dim, N)
        v = self.v_proj(x_norm).view(B, self.num_heads, self.head_dim, N)
        q = self.q_proj(x_norm).view(B, self.num_heads, self.head_dim, N)

        # --- Force float32 for linear attention accumulation ---
        # Under AMP, autocast recasts float32 einsum inputs to float16,
        # causing overflow when summing over N spatial positions (up to 9216
        # at p2 level). Disable autocast entirely for this block.
        orig_dtype = k.dtype
        with torch.amp.autocast("cuda", enabled=False):
            k = k.float()
            v = v.float()
            q = q.float()

            # Positive feature map: phi(·) = ELU(·) + 1
            phi_k = F.elu(k) + 1.0  # (B, H_heads, D, N)
            phi_q = F.elu(q) + 1.0

            # Accumulate memory: M = M_0 + Σ_n phi(k_n) ⊗ v_n^T
            # M shape: (B, H_heads, D, D)
            M = torch.einsum("bhdn,bhen->bhde", phi_k, v) + self.M_0.float().unsqueeze(0)

            # Normalization: Z = Z_0 + Σ_n phi(k_n)
            Z = phi_k.sum(dim=-1) + self.Z_0.float().unsqueeze(0)  # (B, H_heads, D)

            # Retrieve: y_n = M^T @ phi(q_n) / (Z^T @ phi(q_n))
            y_num = torch.einsum("bhde,bhdn->bhen", M, phi_q)  # (B, H, D, N)
            y_den = torch.einsum("bhd,bhdn->bhn", Z, phi_q)    # (B, H, N)
            y_den = y_den.unsqueeze(2).clamp(min=1e-4)          # (B, H, 1, N)
            y = y_num / y_den  # (B, H_heads, D, N)

        # Cast back to original dtype (float16 under AMP)
        y = y.to(dtype=orig_dtype)

        # Reshape back to spatial
        y = y.reshape(B, self.memory_dim, H, W)
        y = self.out_proj(y)

        # Gated residual: output = x + gate * memory_output
        gate = torch.sigmoid(self.gate_net(x_norm)).view(B, 1, 1, 1)

        return residual + gate * y, gate


# =============================================================================
# B) Prototype Memory Bank — Dual-speed EMA in feature space
# =============================================================================

class PrototypeMemoryBank(nn.Module):
    """
    Prototype memory bank with dual-speed EMA, operating in feature space.

    Moved from SafeNestedResidualRefiner into the decoder. Prototypes accumulate
    cross-sample knowledge during training via EMA updates.

    NL mapping (§7 CMS):
        Level 2c: Prototype banks are updated at f=1/batch via EMA
        Two timescales: fast bank (momentum=0.03) captures recent patterns,
                        slow bank (momentum=0.0075) captures stable priors

    Key difference from Refiner version:
        - Operates on feature space (B, C, H, W) instead of logit space
        - Token = spatial average pooling (no uncertainty-weighted pooling since
          logits don't exist yet at this stage)
        - Output = prototype context vector for augmenting fused features

    Args:
        feature_dim: input feature channels (fpn_channels)
        prototype_dim: dimension of prototype space
        num_prototypes: K prototypes per bank
        fast_momentum: EMA momentum for fast bank
        slow_momentum: EMA momentum for slow bank
        max_norm: L2 norm clamp for prototypes
    """

    def __init__(
        self,
        feature_dim: int,
        prototype_dim: int = 128,
        num_prototypes: int = 8,
        fast_momentum: float = 0.03,
        slow_momentum: float = 0.0075,
        max_norm: float = 1.0,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.fast_momentum = fast_momentum
        self.slow_momentum = slow_momentum
        self.max_norm = max_norm

        # Project features into prototype space
        self.query_proj = nn.Conv2d(feature_dim, prototype_dim, 1, bias=False)

        # Prototype banks as buffers (NOT parameters — updated via EMA only)
        self.register_buffer("fast_prototypes", torch.zeros(num_prototypes, prototype_dim))
        self.register_buffer("slow_prototypes", torch.zeros(num_prototypes, prototype_dim))
        self.register_buffer("fast_counts", torch.zeros(num_prototypes))
        self.register_buffer("slow_counts", torch.zeros(num_prototypes))

        # Learnable mix gate between fast and slow banks
        self.mix_gate = nn.Sequential(
            nn.Linear(prototype_dim, prototype_dim // 2),
            nn.GELU(),
            nn.Linear(prototype_dim // 2, 1),
            nn.Sigmoid(),
        )

    def _retrieve(
        self, tokens: Tensor, bank: Tensor, counts: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Scaled dot-product attention retrieval from a prototype bank.

        Args:
            tokens: (B, D) L2-normalized query tokens
            bank: (K, D) prototype bank
            counts: (K,) usage counts

        Returns:
            context: (B, D) retrieved prototype context
            attn: (B, K) attention weights (zero for inactive slots)
        """
        device, dtype = tokens.device, tokens.dtype
        ready = (counts > 1e-6).to(device=device)

        if not bool(ready.any()):
            return (
                torch.zeros_like(tokens),
                torch.zeros(tokens.size(0), bank.size(0), device=device, dtype=dtype),
            )

        # Disable autocast: F.normalize on near-zero fp16 → NaN, matmul may overflow
        with torch.amp.autocast("cuda", enabled=False):
            bank_ready = F.normalize(bank[ready].to(device=device).float(), dim=-1)
            tokens_f32 = tokens.float()
            logits = torch.matmul(tokens_f32, bank_ready.t()) / math.sqrt(self.prototype_dim)
            attn_ready = torch.softmax(logits, dim=-1)
            context = torch.matmul(attn_ready, bank_ready).to(dtype=dtype)

        attn_full = torch.zeros(tokens.size(0), bank.size(0), device=device, dtype=dtype)
        attn_full[:, ready] = attn_ready.to(dtype=dtype)
        return context, attn_full

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        """
        Retrieve prototype context for fused features.

        Args:
            features: (B, C, H, W) fused feature map from BiFPN

        Returns:
            dict with:
                context_spatial: (B, prototype_dim, H, W) broadcast context
                context_global: (B, prototype_dim) global context vector
                token: (B, prototype_dim) detached token for EMA update
                attn_fast/attn_slow: (B, K) detached attention weights
                mix: (B, 1) detached mix gate value
        """
        B, _, H, W = features.shape

        # Spatial average pooling → token (no logits → no uncertainty weighting)
        proj = self.query_proj(features)                     # (B, D, H, W)
        token = F.adaptive_avg_pool2d(proj, 1).flatten(1)    # (B, D)
        # Normalize in float32 to avoid NaN when token is near-zero in fp16
        token = F.normalize(token.float(), dim=-1).to(dtype=features.dtype)

        # Retrieve from both banks
        ctx_fast, attn_fast = self._retrieve(token, self.fast_prototypes, self.fast_counts)
        ctx_slow, attn_slow = self._retrieve(token, self.slow_prototypes, self.slow_counts)

        # Learned mix between fast (recent patterns) and slow (stable priors)
        mix = self.mix_gate(token.float()).to(dtype=features.dtype)  # (B, 1)
        context = mix * ctx_fast + (1.0 - mix) * ctx_slow  # (B, D)

        # Broadcast to spatial dimensions for fusion
        context_spatial = context.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        return {
            "context_spatial": context_spatial,
            "context_global": context,
            "token": token.detach(),
            "attn_fast": attn_fast.detach(),
            "attn_slow": attn_slow.detach(),
            "mix": mix.detach(),
        }

    @torch.no_grad()
    def update_prototypes(self, cache: Dict[str, Tensor]) -> None:
        """
        EMA update of prototype banks after each training batch.

        Called externally by the training loop (same pattern as
        SafeNestedResidualRefiner.update_prototypes).

        Args:
            cache: dict from forward() containing 'token', 'attn_fast', 'attn_slow'
        """
        if cache is None:
            return
        tokens = cache["token"].to(device=self.fast_prototypes.device,
                                   dtype=self.fast_prototypes.dtype)
        attn_fast = cache["attn_fast"].to(device=self.fast_prototypes.device,
                                          dtype=self.fast_prototypes.dtype)
        attn_slow = cache["attn_slow"].to(device=self.slow_prototypes.device,
                                          dtype=self.slow_prototypes.dtype)

        self._update_bank(self.fast_prototypes, self.fast_counts,
                          tokens, attn_fast, self.fast_momentum)
        self._update_bank(self.slow_prototypes, self.slow_counts,
                          tokens, attn_slow, self.slow_momentum)

    @torch.no_grad()
    def _update_bank(
        self, bank: Tensor, counts: Tensor,
        tokens: Tensor, attn: Tensor, momentum: float,
    ) -> None:
        """Bootstrap empty slots then EMA-update active prototypes."""
        B = tokens.size(0)
        tokens_norm = F.normalize(tokens, dim=-1)

        # Bootstrap: fill inactive slots with diverse tokens
        inactive = torch.nonzero(counts <= 1e-6, as_tuple=False).flatten()
        if inactive.numel() > 0 and B > 0:
            num_boot = min(inactive.numel(), B)
            # Pick tokens with highest attention spread (most informative)
            for i in range(num_boot):
                bank[inactive[i]].copy_(tokens_norm[i % B])
                counts[inactive[i]].fill_(1.0)

        # EMA update: assign each token to best-matching prototype
        active = torch.nonzero(counts > 1e-6, as_tuple=False).flatten()
        if active.numel() == 0:
            return

        bank_active = F.normalize(bank[active], dim=-1)
        sim = torch.matmul(tokens_norm, bank_active.t())  # (B, K_active)
        assignments = sim.argmax(dim=-1)                   # (B,)

        for k_local in range(active.numel()):
            k_global = int(active[k_local].item())
            mask = assignments == k_local
            if not bool(mask.any()):
                continue
            target = F.normalize(tokens_norm[mask].mean(dim=0), dim=0)
            bank[k_global].mul_(1.0 - momentum).add_(momentum * target)
            counts[k_global].add_(float(mask.sum().item()))

        # Max-norm constraint
        norms = bank.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.max_norm / (norms + 1e-6), max=1.0)
        bank.mul_(scale)


# =============================================================================
# C) Uncertainty-Aware Fusion — variance-based uncertainty gate
# =============================================================================

class UncertaintyAwareFusion(nn.Module):
    """
    Uncertainty estimation from feature disagreement across FPN levels,
    used to gate prototype augmentation on fused features.

    Replaces the logit-based uncertainty gate of SafeNestedResidualRefiner.
    Since we operate in feature space (before seg_head), uncertainty is estimated
    from cross-level feature variance: regions where FPN levels disagree have
    high uncertainty and benefit most from prototype augmentation.

    Formulation:
        variance = Var(upsampled_level_features)   across levels
        Γ_unc = sigmoid(MLP(variance))
        output = fused_feat + Γ_unc * project(prototype_context)

    NL mapping: this is the "uncertainty gate" component that controls how much
    prototype knowledge (Level 2c) influences the final features.

    Args:
        channels: fused feature channels (fpn_channels)
        proto_dim: prototype context channels
    """

    def __init__(self, channels: int, proto_dim: int):
        super().__init__()
        # Use GroupNorm instead of BatchNorm to avoid train/eval discrepancy.
        # BatchNorm accumulates running stats from zero-input prototype contexts
        # during pre-nested epochs; when prototypes activate, eval uses stale
        # running stats → extreme outputs → validation collapse.
        self.unc_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, max(1, channels // 8)), channels // 2),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 1),
        )
        # Start very conservative — sigmoid(-3) ≈ 0.05 (was -1.0 ≈ 0.27)
        nn.init.constant_(self.unc_head[-1].bias, -3.0)

        # Project prototype context to feature space
        self.context_proj = nn.Sequential(
            nn.Conv2d(proto_dim, channels, 1, bias=False),
            nn.GroupNorm(min(8, max(1, channels // 4)), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, max(1, channels // 4)), channels),
        )

    def forward(
        self,
        fused: Tensor,
        level_features: List[Tensor],
        proto_context_spatial: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Uncertainty-gated prototype augmentation.

        Args:
            fused: (B, C, H, W) fused BiFPN features
            level_features: list of (B, C, H, W) per-level features (already
                            upsampled to fused resolution)
            proto_context_spatial: (B, proto_dim, H, W) prototype context

        Returns:
            augmented: (B, C, H, W) fused features + gated prototype context
            unc_map: (B, 1, H, W) uncertainty map for logging
        """
        # Cross-level variance → uncertainty estimate
        # Disable autocast: var() involves squaring + accumulation, overflows fp16
        with torch.amp.autocast("cuda", enabled=False):
            stacked = torch.stack(level_features, dim=0).float()  # (L, B, C, H, W)
            variance = stacked.var(dim=0).to(dtype=fused.dtype)   # (B, C, H, W)

        unc_map = torch.sigmoid(self.unc_head(variance))  # (B, 1, H, W)

        # Project prototype context to match feature channels
        context = self.context_proj(proto_context_spatial)  # (B, C, H, W)

        # Uncertainty-gated augmentation: augment more where uncertain
        augmented = fused + unc_map * context
        return augmented, unc_map


# =============================================================================
# MAIN: CMSDecoder — BiFPN + CMS Memory + Prototype + Uncertainty
# =============================================================================

class CMSDecoder(nn.Module):
    """
    BiFPN Decoder + Continuum Memory System + Prototype Memory.

    Replaces: FPNDecoder + SafeNestedResidualRefiner

    Three main components:

    A) BiFPN Feature Fusion (preserves existing BiFPN logic)
       - SE channel attention on lateral projections
       - Top-down pathway with learnable weighted fusion
       - Bottom-up pathway with learnable weighted fusion

    B) CMS Memory Modules (NL Strategy A, §7)
       - One CMSMemoryModule per selected FPN level
       - Each level has linear attention memory with meta-learned M₀
       - Update frequency varies naturally by spatial resolution (CMS):
         * p2 (1/4): fast memory, dense spatial details
         * p3 (1/8): medium memory
         * p4 (1/16): slow memory, semantic
         * p5 (1/32): slowest, global context

    C) Prototype-Augmented Fusion (replaces Refiner)
       - Dual-speed EMA prototype banks (fast + slow)
       - Uncertainty-aware gating from cross-level feature variance
       - Result: fused features augmented by cross-sample prototype knowledge

    Backward compatibility: setting cms_levels=[] and num_prototypes=0 makes
    this behave like a plain BiFPN decoder.

    Args:
        encoder_channels: channel dims for [c2, c3, c4, c5] from encoder
        fpn_channels: unified FPN channel dimension
        num_prototypes: K prototypes per bank (0 to disable)
        prototype_dim: dimension of prototype space
        fast_momentum: EMA momentum for fast prototype bank
        slow_momentum: EMA momentum for slow prototype bank
        num_heads: multi-head count for CMS memory
        cms_levels: which FPN levels get CMS memory ([] to disable)
        memory_dim: dimension for CMS linear attention
        gate_init_bias: initial gate bias for CMS modules
        use_hierarchical_prototypes: bool, use hierarchical tree-structured prototypes
        hierarchical_levels: number of hierarchy levels (if use_hierarchical=True)
        prototypes_per_level: prototypes per level (int or list)
    """

    def __init__(
        self,
        encoder_channels: List[int] = [64, 128, 320, 512],
        fpn_channels: int = 128,
        num_prototypes: int = 8,
        prototype_dim: int = 128,
        fast_momentum: float = 0.03,
        slow_momentum: float = 0.0075,
        num_heads: int = 4,
        cms_levels: List[int] = [0, 1, 2, 3],
        memory_dim: int = 64,
        gate_init_bias: float = -2.0,
        use_hierarchical_prototypes: bool = False,
        hierarchical_levels: int = 3,
        prototypes_per_level: Union[int, List[int]] = 8,
    ):
        super().__init__()
        self.fpn_channels = fpn_channels
        self.cms_levels = list(cms_levels)
        self.num_prototypes = num_prototypes
        c2_ch, c3_ch, c4_ch, c5_ch = encoder_channels

        # --- BiFPN: Lateral projections + SE attention ---
        self.lateral5 = nn.Conv2d(c5_ch, fpn_channels, 1)
        self.lateral4 = nn.Conv2d(c4_ch, fpn_channels, 1)
        self.lateral3 = nn.Conv2d(c3_ch, fpn_channels, 1)
        self.lateral2 = nn.Conv2d(c2_ch, fpn_channels, 1)

        self.se5 = _SEBlock(fpn_channels)
        self.se4 = _SEBlock(fpn_channels)
        self.se3 = _SEBlock(fpn_channels)
        self.se2 = _SEBlock(fpn_channels)

        # --- BiFPN: Top-down weighted fusion ---
        self.bifpn_td4 = _BiFPNFuseCell(fpn_channels)
        self.bifpn_td3 = _BiFPNFuseCell(fpn_channels)
        self.bifpn_td2 = _BiFPNFuseCell(fpn_channels)

        # --- BiFPN: Bottom-up weighted fusion ---
        self.bifpn_bu3 = _BiFPNFuseCell(fpn_channels)
        self.bifpn_bu4 = _BiFPNFuseCell(fpn_channels)
        self.bifpn_bu5 = _BiFPNFuseCell(fpn_channels)

        # --- CMS Memory Modules (one per selected FPN level) ---
        self.cms_modules = nn.ModuleDict()
        level_names = ["p2", "p3", "p4", "p5"]
        for lvl in self.cms_levels:
            self.cms_modules[level_names[lvl]] = CMSMemoryModule(
                channels=fpn_channels,
                memory_dim=memory_dim,
                num_heads=num_heads,
                gate_bias=gate_init_bias,
            )

        # --- Smoothing + Fusion ---
        self.smooth5 = _ConvBNAct(fpn_channels, fpn_channels)
        self.smooth4 = _ConvBNAct(fpn_channels, fpn_channels)
        self.smooth3 = _ConvBNAct(fpn_channels, fpn_channels)
        self.smooth2 = _ConvBNAct(fpn_channels, fpn_channels)
        self.fuse = nn.Sequential(
            _ConvBNAct(fpn_channels * 4, fpn_channels),
            _ConvBNAct(fpn_channels, fpn_channels),
        )

        # --- Prototype Memory Bank ---
        self.prototype = None
        self.use_hierarchical_prototypes = use_hierarchical_prototypes
        if num_prototypes > 0:
            if use_hierarchical_prototypes:
                from model.advanced_modules import HierarchicalPrototypeBank
                self.prototype = HierarchicalPrototypeBank(
                    feature_dim=fpn_channels,
                    num_levels=hierarchical_levels,
                    prototypes_per_level=prototypes_per_level,
                    fast_momentum=fast_momentum,
                    slow_momentum=slow_momentum,
                )
            else:
                self.prototype = PrototypeMemoryBank(
                    feature_dim=fpn_channels,
                    prototype_dim=prototype_dim,
                    num_prototypes=num_prototypes,
                    fast_momentum=fast_momentum,
                    slow_momentum=slow_momentum,
                )

        # --- Uncertainty-Aware Fusion ---
        self.uncertainty_fusion = None
        if num_prototypes > 0:
            self.uncertainty_fusion = UncertaintyAwareFusion(
                channels=fpn_channels,
                proto_dim=prototype_dim,
            )

    def forward(
        self,
        features: List[Tensor],
        gt_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        Full decoder forward pass: BiFPN → CMS memory → prototype → uncertainty fusion.

        Args:
            features: [c2, c3, c4, c5] feature maps from encoder
            gt_mask: ground truth mask (reserved for future use; currently unused)

        Returns:
            fused_feat: (B, fpn_channels, H/4, W/4) adapted features for seg_head
            aux_feat: (B, encoder_channels[-1], H/32, W/32) for aux_head
            decoder_info: dict containing:
                - uncertainty_map: (B, 1, H/4, W/4)
                - prototype_sim: prototype similarity info
                - cms_gate_values: gate values per FPN level
                - memory_states: debug info
        """
        c2, c3, c4, c5 = features[0], features[1], features[2], features[3]

        # === BiFPN: Lateral + SE channel attention ===
        p5 = self.se5(self.lateral5(c5))
        p4 = self.se4(self.lateral4(c4))
        p3 = self.se3(self.lateral3(c3))
        p2 = self.se2(self.lateral2(c2))

        # === BiFPN: Top-down pathway ===
        p4_td = self.bifpn_td4(
            p4, F.interpolate(p5, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        )
        p3_td = self.bifpn_td3(
            p3, F.interpolate(p4_td, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        )
        p2_td = self.bifpn_td2(
            p2, F.interpolate(p3_td, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        )

        # === BiFPN: Bottom-up pathway ===
        p3_out = self.bifpn_bu3(
            p3_td, F.interpolate(p2_td, size=p3_td.shape[-2:], mode="bilinear", align_corners=False)
        )
        p4_out = self.bifpn_bu4(
            p4_td, F.interpolate(p3_out, size=p4_td.shape[-2:], mode="bilinear", align_corners=False)
        )
        p5_out = self.bifpn_bu5(
            p5, F.interpolate(p4_out, size=p5.shape[-2:], mode="bilinear", align_corners=False)
        )

        # === CMS Memory: apply on each level's output ===
        level_names = ["p2", "p3", "p4", "p5"]
        level_outputs = [p2_td, p3_out, p4_out, p5_out]
        cms_gate_values = {}

        for lvl in self.cms_levels:
            name = level_names[lvl]
            if name in self.cms_modules:
                level_outputs[lvl], gate_val = self.cms_modules[name](level_outputs[lvl])
                cms_gate_values[name] = gate_val.detach().mean()

        p2_out_final, p3_out_final, p4_out_final, p5_out_final = level_outputs

        # === Smooth each level ===
        s2 = self.smooth2(p2_out_final)
        s3 = self.smooth3(p3_out_final)
        s4 = self.smooth4(p4_out_final)
        s5 = self.smooth5(p5_out_final)

        # === Upsample to p2 resolution and fuse ===
        target_size = s2.shape[-2:]
        s3_up = F.interpolate(s3, size=target_size, mode="bilinear", align_corners=False)
        s4_up = F.interpolate(s4, size=target_size, mode="bilinear", align_corners=False)
        s5_up = F.interpolate(s5, size=target_size, mode="bilinear", align_corners=False)

        fused = self.fuse(torch.cat([s2, s3_up, s4_up, s5_up], dim=1))

        # === Prototype Memory + Uncertainty-Aware Fusion ===
        decoder_info: Dict[str, object] = {
            "cms_gate_values": cms_gate_values,
        }
        proto_cache = None

        if self.prototype is not None and self.uncertainty_fusion is not None:
            proto_out = self.prototype(fused)
            if self.use_hierarchical_prototypes:
                # Hierarchical bank returns per-level attention
                proto_cache = {
                    "token": proto_out["token"],
                    "attn_per_level": proto_out["attn_per_level"],
                    "level_weights": proto_out["level_weights"],
                    "level_mix": proto_out["level_mix"],
                }
                decoder_info["prototype_level_weights"] = proto_out["level_weights"].detach()
            else:
                # Standard dual-speed bank
                proto_cache = {
                    "token": proto_out["token"],
                    "attn_fast": proto_out["attn_fast"],
                    "attn_slow": proto_out["attn_slow"],
                }
                decoder_info["prototype_mix"] = proto_out["mix"].mean()
                decoder_info["prototype_sim_fast"] = proto_out["attn_fast"]
                decoder_info["prototype_sim_slow"] = proto_out["attn_slow"]

            # Uncertainty-gated augmentation
            fused, unc_map = self.uncertainty_fusion(
                fused,
                [s2, s3_up, s4_up, s5_up],
                proto_out["context_spatial"],
            )

            decoder_info["uncertainty_map"] = unc_map.detach()
        else:
            decoder_info["uncertainty_map"] = torch.zeros(
                fused.size(0), 1, *fused.shape[-2:],
                device=fused.device, dtype=fused.dtype,
            )

        decoder_info["proto_cache"] = proto_cache

        # aux_feat = s5 from BiFPN (for auxiliary head)
        aux_feat = s5

        # Package results as dict for flexibility
        result = {
            'fused': fused,
            'aux_feat': aux_feat,
            'smoothed_features': [s2, s3, s4, s5],  # at native resolutions
            'decoder_info': decoder_info
        }
        return result

    @torch.no_grad()
    def update_prototypes(self, decoder_info: Dict) -> None:
        """
        Update prototype banks via EMA. Call after each training batch.

        Args:
            decoder_info: dict returned by forward(), must contain 'proto_cache'
        """
        if self.prototype is None:
            return
        cache = decoder_info.get("proto_cache")
        if cache is not None:
            self.prototype.update_prototypes(cache)


# =============================================================================
# SMOKE TEST
# =============================================================================

def _smoke_test():
    """
    Verify: shapes, gradient flow, eval mode, prototype EMA update,
    backward compatibility (cms_levels=[], num_prototypes=0).
    """
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, W = 2, 384, 384
    encoder_channels = [64, 128, 320, 512]

    # Dummy encoder features
    c2 = torch.randn(B, 64, H // 4, H // 4, device=device, requires_grad=True)
    c3 = torch.randn(B, 128, H // 8, H // 8, device=device, requires_grad=True)
    c4 = torch.randn(B, 320, H // 16, H // 16, device=device, requires_grad=True)
    c5 = torch.randn(B, 512, H // 32, H // 32, device=device, requires_grad=True)
    features = [c2, c3, c4, c5]

    print("=" * 60)
    print("CMS Decoder Smoke Test")
    print("=" * 60)

    # --- Test 1: Full CMS Decoder ---
    print("\n[1] Full CMSDecoder (all CMS levels + prototypes)...")
    decoder = CMSDecoder(
        encoder_channels=encoder_channels,
        fpn_channels=128,
        num_prototypes=8,
        prototype_dim=128,
        cms_levels=[0, 1, 2, 3],
        memory_dim=64,
        num_heads=4,
    ).to(device)

    result = decoder(features)
    fused = result['fused']
    aux = result['aux_feat']
    info = result['decoder_info']
    assert fused.shape == (B, 128, H // 4, H // 4), f"fused shape: {fused.shape}"
    assert aux.shape == (B, 128, H // 32, H // 32), f"aux shape: {aux.shape}"  # aux = s5 after BiFPN smoothing (128-ch)
    assert "uncertainty_map" in info
    assert info["uncertainty_map"].shape == (B, 1, H // 4, H // 4)
    assert "cms_gate_values" in info
    assert len(info["cms_gate_values"]) == 4
    # Check smoothed_features exist
    assert "smoothed_features" in result
    smoothed = result["smoothed_features"]
    assert len(smoothed) == 4
    assert smoothed[0].shape == (B, 128, H // 4, H // 4)  # s2
    assert smoothed[1].shape == (B, 128, H // 8, H // 8)  # s3
    assert smoothed[2].shape == (B, 128, H // 16, H // 16)  # s4
    assert smoothed[3].shape == (B, 128, H // 32, H // 32)  # s5
    print(f"   fused: {fused.shape}, aux: {aux.shape}")
    print(f"   uncertainty_map: {info['uncertainty_map'].shape}")
    print(f"   smoothed_features: s2={tuple(smoothed[0].shape)}, s3={tuple(smoothed[1].shape)}")
    print(f"   CMS gates: { {k: f'{v.item():.4f}' for k, v in info['cms_gate_values'].items()} }")
    print("   PASS")

    # --- Test 2: Gradient flow ---
    print("\n[2] Gradient flow...")
    loss = fused.sum() + aux.sum()
    loss.backward()
    assert c2.grad is not None and c2.grad.abs().sum() > 0, "No gradient to c2"
    assert c5.grad is not None and c5.grad.abs().sum() > 0, "No gradient to c5"
    # Check CMS module gradients
    for name, param in decoder.cms_modules.named_parameters():
        assert param.grad is not None, f"No gradient for CMS param: {name}"
    # Check prototype query_proj gradient
    for name, param in decoder.prototype.named_parameters():
        assert param.grad is not None, f"No gradient for prototype param: {name}"
    print("   All gradients OK")
    print("   PASS")

    # --- Test 3: Prototype EMA update ---
    print("\n[3] Prototype EMA update...")
    old_fast = decoder.prototype.fast_prototypes.clone()
    old_counts = decoder.prototype.fast_counts.clone()
    decoder.update_prototypes(info)
    new_counts = decoder.prototype.fast_counts
    assert new_counts.sum() > old_counts.sum(), "Prototype counts did not increase"
    print(f"   fast_counts: {old_counts.sum().item():.1f} -> {new_counts.sum().item():.1f}")
    print("   PASS")

    # --- Test 4: Eval mode ---
    print("\n[4] Eval mode forward...")
    decoder.eval()
    with torch.no_grad():
        c2_eval = torch.randn(1, 64, H // 4, H // 4, device=device)
        c3_eval = torch.randn(1, 128, H // 8, H // 8, device=device)
        c4_eval = torch.randn(1, 320, H // 16, H // 16, device=device)
        c5_eval = torch.randn(1, 512, H // 32, H // 32, device=device)
        result_eval = decoder([c2_eval, c3_eval, c4_eval, c5_eval])
        fused_eval = result_eval['fused']
        aux_eval = result_eval['aux_feat']
        info_eval = result_eval['decoder_info']
    assert fused_eval.shape == (1, 128, H // 4, H // 4)
    print(f"   fused: {fused_eval.shape}")
    print("   PASS")
    decoder.train()

    # --- Test 5: Backward compatibility (disable CMS + prototype) ---
    print("\n[5] Backward compat (cms_levels=[], num_prototypes=0)...")
    decoder_plain = CMSDecoder(
        encoder_channels=encoder_channels,
        fpn_channels=128,
        num_prototypes=0,
        cms_levels=[],
    ).to(device)

    c2_p = torch.randn(B, 64, H // 4, H // 4, device=device, requires_grad=True)
    c3_p = torch.randn(B, 128, H // 8, H // 8, device=device, requires_grad=True)
    c4_p = torch.randn(B, 320, H // 16, H // 16, device=device, requires_grad=True)
    c5_p = torch.randn(B, 512, H // 32, H // 32, device=device, requires_grad=True)

    result_p = decoder_plain([c2_p, c3_p, c4_p, c5_p])
    fused_p = result_p['fused']
    aux_p = result_p['aux_feat']
    info_p = result_p['decoder_info']
    assert fused_p.shape == (B, 128, H // 4, H // 4)
    assert len(info_p["cms_gate_values"]) == 0
    assert info_p["proto_cache"] is None
    loss_p = fused_p.sum()
    loss_p.backward()
    assert c2_p.grad is not None
    print(f"   fused: {fused_p.shape}, cms_gates: {{}}, proto_cache: None")
    print("   PASS")

    # --- Test 6: Parameter count ---
    print("\n[6] Parameter count...")
    total = sum(p.numel() for p in decoder.parameters())
    trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in decoder.buffers())
    print(f"   Total params:     {total:,}")
    print(f"   Trainable params: {trainable:,}")
    print(f"   Buffers:          {buffers:,}")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _smoke_test()
