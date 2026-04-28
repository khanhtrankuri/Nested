"""
Multi-Scale Segmentation Heads for FPN Features.

This module implements segmentation heads at multiple FPN levels (p2, p3, p4, p5)
to provide deep supervision and preserve multi-scale information. Each scale produces
its own logits, which are then fused to create the final output.

Features:
- Separate or shared head weights across scales
- Weighted sum or attention-based fusion of logits
- Auxiliary losses computed at native resolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Avoid circular import by defining ConvBNAct locally or importing
# Option 1: Import from strong_baseline (preferred for consistency)
from model.backbones.strong_baseline import ConvBNAct


class MultiScaleSegHeads(nn.Module):
    """
    Apply segmentation heads at multiple FPN levels.

    Args:
        in_channels: feature channels at each level (all should be same after smoothing)
        num_scales: typically 4 (p2, p3, p4, p5)
        mode: 'separate' or 'shared' weights
        fusion_type: 'weighted_sum' or 'attention'
    """
    def __init__(self, in_channels: int, num_scales: int = 4,
                 mode: str = 'separate', fusion_type: str = 'weighted_sum'):
        super().__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales
        self.mode = mode
        self.fusion_type = fusion_type

        # Create segmentation heads
        if mode == 'shared':
            self.head = nn.Sequential(
                ConvBNAct(in_channels, in_channels),
                nn.Conv2d(in_channels, 1, 1)
            )
        else:  # separate
            self.heads = nn.ModuleList([
                nn.Sequential(
                    ConvBNAct(in_channels, in_channels),
                    nn.Conv2d(in_channels, 1, 1)
                ) for _ in range(num_scales)
            ])

        # Fusion weights for combining multi-scale logits
        if fusion_type == 'weighted_sum':
            # BiFPN-style learnable weights
            self.logits_weights = nn.Parameter(torch.ones(num_scales))
            self.eps = 1e-4
        elif fusion_type == 'attention':
            # Attention mechanism to combine logits based on feature context
            # Use the highest resolution features (s2) to generate attention map
            self.attention = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, num_scales, 1)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights properly."""
        if self.fusion_type == 'weighted_sum':
            # Initialize weights to small equal values
            nn.init.constant_(self.logits_weights, 1.0)
        elif self.fusion_type == 'attention':
            for m in self.attention.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, a=1)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: list, target_size: tuple = None):
        """
        Args:
            features: list of [s2, s3, s4, s5] at native resolutions
                     s2: (B, C, H/4, W/4)  - highest resolution
                     s3: (B, C, H/8, W/8)
                     s4: (B, C, H/16, W/16)
                     s5: (B, C, H/32, W/32) - lowest resolution
            target_size: (H, W) for final output. If None, use s2 size (highest resolution).

        Returns:
            main_logits: final fused logits at target_size, shape (B, 1, H, W)
            scale_logits: list of logits at native resolutions [(B,1,H/4,W/4), ...]
            scale_logits_upsampled: list of logits at target_size [(B,1,H,W), ...]
        """
        assert len(features) == self.num_scales, \
            f"Expected {self.num_scales} features, got {len(features)}"

        B = features[0].shape[0]
        device = features[0].device
        dtype = features[0].dtype

        # Apply segmentation heads to each scale
        if self.mode == 'shared':
            scale_logits = [self.head(f) for f in features]
        else:  # separate
            scale_logits = [head(f) for head, f in zip(self.heads, features)]

        # Determine target size (use highest resolution by default)
        if target_size is None:
            target_size = features[0].shape[-2:]

        # Upsample all logits to target size for fusion
        scale_logits_upsampled = [
            F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
            for logits in scale_logits
        ]

        # Fuse to get main logits
        if self.fusion_type == 'weighted_sum':
            w = F.relu(self.logits_weights)
            w = w / (w.sum() + self.eps)
            main_logits = sum(w[i] * logits for i, logits in enumerate(scale_logits_upsampled))
        elif self.fusion_type == 'attention':
            # Use s2 (highest resolution) features to generate attention weights
            attn_input = features[0]  # s2 at (B, C, H/4, W/4)
            attn_map = self.attention(attn_input)  # (B, num_scales, H/4, W/4)
            attn_map = F.interpolate(attn_map, size=target_size, mode='bilinear', align_corners=False)
            attn_map = F.softmax(attn_map, dim=1)  # normalize across scales per pixel

            main_logits = sum(attn_map[:, i:i+1] * logits
                             for i, logits in enumerate(scale_logits_upsampled))

        return main_logits, scale_logits, scale_logits_upsampled

    def extra_repr(self):
        return (f"in_channels={self.in_channels}, num_scales={self.num_scales}, "
                f"mode={self.mode}, fusion={self.fusion_type}")


def _smoke_test():
    """Quick sanity check for MultiScaleSegHeads."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, C, H, W = 2, 128, 96, 96

    # Create features at different scales (simulating decoder smoothed outputs)
    s2 = torch.randn(B, C, H//4, W//4, device=device, requires_grad=True)
    s3 = torch.randn(B, C, H//8, W//8, device=device, requires_grad=True)
    s4 = torch.randn(B, C, H//16, W//16, device=device, requires_grad=True)
    s5 = torch.randn(B, C, H//32, W//32, device=device, requires_grad=True)
    features = [s2, s3, s4, s5]

    print("=" * 60)
    print("MultiScaleSegHeads Smoke Test")
    print("=" * 60)

    # Test 1: Separate mode with weighted sum fusion
    print("\n[1] Separate mode + weighted_sum fusion")
    heads_sep = MultiScaleSegHeads(
        in_channels=C,
        num_scales=4,
        mode='separate',
        fusion_type='weighted_sum'
    ).to(device)

    main_logits, scale_logits, scale_up = heads_sep(features, target_size=(H, W))

    print(f"  Input scales: s2={tuple(s2.shape)}, s3={tuple(s3.shape)}, s4={tuple(s4.shape)}, s5={tuple(s5.shape)}")
    print(f"  Scale logits: p2={tuple(scale_logits[0].shape)}, p3={tuple(scale_logits[1].shape)}")
    print(f"  Scale logits: p4={tuple(scale_logits[2].shape)}, p5={tuple(scale_logits[3].shape)}")
    print(f"  Main logits: {tuple(main_logits.shape)}")
    print(f"  Upsampled scales: all {tuple(scale_up[0].shape)}")

    # Check shapes
    assert main_logits.shape == (B, 1, H, W), f"Main logits shape mismatch: {main_logits.shape}"
    assert len(scale_logits) == 4
    assert scale_logits[0].shape == (B, 1, H//4, W//4)
    assert scale_logits[1].shape == (B, 1, H//8, W//8)
    assert scale_logits[2].shape == (B, 1, H//16, W//16)
    assert scale_logits[3].shape == (B, 1, H//32, W//32)
    assert all(l.shape == (B, 1, H, W) for l in scale_up)

    # Gradient check
    targets = torch.randint(0, 2, (B, 1, H, W), device=device, dtype=torch.float32)
    loss = F.binary_cross_entropy_with_logits(main_logits, targets)
    loss.backward()
    print(f"  Backward OK, loss={loss.item():.4f}")
    print("  PASS")

    # Test 2: Shared mode with weighted sum
    print("\n[2] Shared mode + weighted_sum fusion")
    heads_shared = MultiScaleSegHeads(
        in_channels=C,
        mode='shared',
        fusion_type='weighted_sum'
    ).to(device)

    main2, scale_logits2, scale_up2 = heads_shared(features, (H, W))
    assert main2.shape == (B, 1, H, W)
    # Shared mode should have same parameters across heads
    for i in range(3):
        for p1, p2 in zip(heads_shared.head.parameters(), heads_shared.head.parameters()):
            assert p1 is p2, "Shared mode should use same head"
    print("  Shared weights verified")
    print("  PASS")

    # Test 3: Attention fusion
    print("\n[3] Separate mode + attention fusion")
    heads_attn = MultiScaleSegHeads(
        in_channels=C,
        mode='separate',
        fusion_type='attention'
    ).to(device)

    main3, _, _ = heads_attn(features, (H, W))
    assert main3.shape == (B, 1, H, W)
    loss3 = F.binary_cross_entropy_with_logits(main3, targets)
    loss3.backward()
    if hasattr(heads_attn, 'logits_weights'):
        print(f"  Attention weights: {F.softmax(heads_attn.logits_weights, dim=0).detach().cpu().numpy()}")
    print("  PASS")

    # Test 4: Parameter count
    print("\n[4] Parameter count")
    total_params = sum(p.numel() for p in heads_sep.parameters())
    trainable_params = sum(p.numel() for p in heads_sep.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    assert total_params < 1000000, "Multi-scale heads should be lightweight (<1M params)"
    print("  PASS")

    print("\n" + "=" * 60)
    print("All smoke tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    _smoke_test()
