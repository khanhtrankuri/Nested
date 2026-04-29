"""
Lightweight integration test: Encoder → Decoder → Multi-Scale Heads → Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.cms_decoder import CMSDecoder
from model.backbones.self_modifying_encoder_minimal import MinimalCMSEncoder
from model.multi_scale_heads import MultiScaleSegHeads
from loss.strong_baseline_loss import StrongBaselineLoss
from loss.boundary_losses import SoftDiceBoundaryLoss


def test_full_pipeline():
    print("=" * 70)
    print("Integration Test: CMS Decoder + Boundary + Multi-Scale")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Config
    B, C, H, W = 2, 3, 384, 384
    encoder_channels = [64, 128, 320, 512]
    fpn_channels = 128

    x = torch.randn(B, C, H, W, device=device)
    target = torch.randint(0, 2, (B, 1, H, W), device=device).float()

    # --- Dummy backbone ---
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.out_channels = encoder_channels
        def forward(self, x):
            B = x.shape[0]
            return [
                torch.randn(B, 64, H//4, H//4, device=x.device),
                torch.randn(B, 128, H//8, H//8, device=x.device),
                torch.randn(B, 320, H//16, H//16, device=x.device),
                torch.randn(B, 512, H//32, H//32, device=x.device),
            ]

    backbone = DummyBackbone()

    # --- Encoder ---
    print("\n[1] Encoder (MinimalCMSEncoder)")
    encoder = MinimalCMSEncoder(
        backbone=backbone,
        feature_channels=encoder_channels,
        use_llrd=False,
        c3_adaptor_mode="light",
        use_persistent_momentum=False,
    ).to(device)
    features, nl_info = encoder(x, return_nested_info=True)
    print(f"  Features: {[f.shape for f in features]}")
    print(f"  Nested stages: {list(nl_info.keys())}")

    # --- Decoder ---
    print("\n[2] CMS Decoder")
    decoder = CMSDecoder(
        encoder_channels=encoder_channels,
        fpn_channels=fpn_channels,
        num_prototypes=8,
        prototype_dim=128,
        cms_levels=[0, 1, 2, 3],
        memory_dim=64,
        num_heads=4,
    ).to(device)
    decoder_out = decoder(features)
    fused = decoder_out['fused']
    smoothed = decoder_out['smoothed_features']
    decoder_info = decoder_out['decoder_info']
    print(f"  fused: {fused.shape}")
    print(f"  smoothed: {[s.shape for s in smoothed]}")

    if 'boundary' in decoder_info:
        gate = decoder_info['boundary']['gate_mean'].item()
        edge_map = decoder_info['boundary']['edge_map']
        edge_ratio = (edge_map > 0.1).float().mean().item()
        print(f"  boundary_gate: {gate:.4f}, edge_ratio: {edge_ratio:.4f}")

    # --- Multi-Scale Heads ---
    print("\n[3] Multi-Scale Heads (attention fusion)")
    ms_heads = MultiScaleSegHeads(
        in_channels=fpn_channels,
        num_scales=4,
        mode='separate',
        fusion_type='attention',
    ).to(device)
    main_logits, scale_logits, _ = ms_heads(smoothed, target_size=(H, W))
    print(f"  main_logits: {main_logits.shape}")
    print(f"  scale_logits: {[l.shape for l in scale_logits]}")

    # --- Loss (BCE+Dice + Boundary) ---
    print("\n[4] Loss (BCE+Dice + Boundary)")
    base_loss = StrongBaselineLoss()
    b_loss = SoftDiceBoundaryLoss(boundary_weight=2.0)

    # Compute losses
    loss_base, base_components = base_loss(main_logits, target, return_components=True)
    loss_boundary = b_loss(main_logits, target)
    total_loss = loss_base + 0.1 * loss_boundary

    print(f"  base_loss: {loss_base.item():.4f}")
    if 'bce' in base_components and 'dice' in base_components:
        print(f"    BCE: {base_components['bce']:.4f}, Dice: {base_components['dice']:.4f}")
    print(f"  boundary_loss: {loss_boundary.item():.4f}")
    print(f"  total_loss: {total_loss.item():.4f}")

    # --- Backward ---
    print("\n[5] Backward pass")
    total_loss.backward()
    print("  OK")

    # Grad check
    total_params = sum(p.numel() for p in encoder.parameters())
    total_params += sum(p.numel() for p in decoder.parameters())
    total_params += sum(p.numel() for p in ms_heads.parameters())
    trainable_params = sum(p.numel() for p in [*encoder.parameters(), *decoder.parameters(), *ms_heads.parameters()] if p.requires_grad)
    grads = sum(1 for p in [*encoder.parameters(), *decoder.parameters(), *ms_heads.parameters()] if p.grad is not None)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Gradients: {grads} tensors")

    print("\n" + "=" * 70)
    print("INTEGRATION TEST PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_full_pipeline()
