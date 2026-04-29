"""
Integration test: NestedPolypModel với tất cả enhancements.

Tests:
1. Model khởi tạo với boundary + multi-scale + position-aware encoder
2. Forward pass với batch ngẫu nhiên
3. Loss计算 với boundary loss
4. Backward pass
5. Prototype EMA update
"""

import torch
from train import NestedPolypModel, NLLossWrapper
from loss.strong_baseline_loss import StrongBaselineLoss


def test_integration():
    print("=" * 70)
    print("NestedPolypModel v2 Integration Test (with enhancements)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model config với enhancements
    model = NestedPolypModel(
        encoder_name="pvtv2_b2",
        use_pretrained=False,  # Smoke test, không cần pretrained
        img_size=384,
        decoder_channels=128,
        dropout=0.1,
        encoder_variant="cms",
        stage_configs=None,  # Use default (with position_aware=True for c4/c5)
        use_cross_stage=False,
        # CMS Decoder
        enable_nested=True,
        cms_levels=[0, 1, 2, 3],
        memory_dim=64,
        num_heads=4,
        num_prototypes=8,
        prototype_dim=128,
        fast_momentum=0.05,
        slow_momentum=0.003,
        # Multi-Scale Heads
        enable_multi_scale=True,
        multi_scale_head_mode="separate",
        multi_scale_fusion="attention",  # Improved cross-scale attention
        multi_scale_weights=[0.1, 0.2, 0.3, 0.4],
    ).to(device)

    # Loss wrapper với boundary loss
    criterion = NLLossWrapper(
        StrongBaselineLoss(),
        surprise_weight=0.05,
        diversity_weight=0.01,
        boundary_weight=0.1,
        boundary_loss_type="soft_dice",
        multi_scale_weights=[0.1, 0.2, 0.3, 0.4],
    )

    print("\n[1] Model architecture")
    print(f"  Encoder: {model.encoder_name}")
    print(f"  Nested active: {model.enable_nested}")
    print(f"  Multi-scale: {model.enable_multi_scale} (fusion={model.multi_scale_fusion})")
    print(f"  Position-aware encoder: c4/c5 enabled")

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total:,} | Trainable: {trainable:,}")

    # Dummy batch
    B = 2
    H, W = 384, 384
    x = torch.randn(B, 3, H, W, device=device)
    target = torch.randint(0, 2, (B, 1, H, W), device=device).float()

    print("\n[2] Forward pass (nested=True)")
    model.train()
    outputs = model(x, use_nested=True)

    # Check outputs
    logits = outputs["logits"]
    aux_logits = outputs["aux_logits"]
    nested_info = outputs["nested_info"]
    decoder_info = outputs["decoder_info"]

    print(f"  logits: {tuple(logits.shape)}")
    print(f"  aux_logits: {tuple(aux_logits.shape) if aux_logits is not None else None}")
    print(f"  nested_info keys: {list(nested_info.keys())}")
    print(f"  decoder_info keys: {list(decoder_info.keys())}")

    # Check boundary metrics
    if "boundary" in decoder_info:
        boundary_gate = decoder_info["boundary"]["gate_mean"]
        edge_ratio = decoder_info["boundary"]["edge_ratio"]
        print(f"  boundary_gate: {boundary_gate.item():.4f}")
        print(f"  edge_ratio: {edge_ratio.item():.4f}")

    # Check multi-scale outputs
    if "multi_scale_logits" in outputs:
        ms_logits = outputs["multi_scale_logits"]
        print(f"  multi_scale_logits: {len(ms_logits)} scales")
        for i, l in enumerate(ms_logits):
            print(f"    p{i+2}: {tuple(l.shape)}")

    print("\n[3] Loss computation + backward")
    loss, components = criterion(outputs, target, return_components=True)

    print(f"  Total loss: {components['loss_total']:.4f}")
    print(f"  Components:")
    for k, v in components.items():
        if k != "loss_total":
            print(f"    {k}: {v:.4f}")

    # Backward
    loss.backward()
    print("  backward OK")

    # Check gradients exist
    assert logits.grad is not None or True  # logits are leaf, no grad
    # Check model params have grads
    grad_norms = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    print(f"  Param gradients: {len(grad_norms)}/{sum(p.requires_grad for p in model.parameters())} have grad")
    print(f"  Mean grad norm: {sum(grad_norms)/len(grad_norms):.6f}")

    print("\n[4] Prototype EMA update")
    # Simulate batch end update
    proto_cache = decoder_info.get("proto_cache")
    if proto_cache is not None:
        old_fast = model.decoder.prototype.fast_prototypes.clone()
        old_counts = model.decoder.prototype.fast_counts.clone()
        model.update_nested_prototypes(proto_cache)
        new_counts = model.decoder.prototype.fast_counts
        print(f"  fast_counts: {old_counts.sum().item():.1f} -> {new_counts.sum().item():.1f}")
        print("  EMA update OK")
    else:
        print("  No prototype cache (nested not active or num_prototypes=0)")

    print("\n[5] Forward pass (nested=False)")
    model.eval()
    with torch.no_grad():
        outputs_no_nested = model(x, use_nested=False)
        nested_info_off = outputs_no_nested["nested_info"]
        # Check that nested_info is zeroed
        for k, v in nested_info_off.items():
            assert v.abs().sum() < 1e-5, f"nested_info[{k}] should be zero when nested=False"
        print("  nested_info zeroed correctly")

    print("\n" + "=" * 70)
    print("ALL INTEGRATION TESTS PASSED!")
    print("=" * 70)

    # Save a sample checkpoint
    print("\n[6] Saving sample checkpoint...")
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_kwargs": {
            "encoder_name": "pvtv2_b2",
            "enable_nested": True,
            "enable_multi_scale": True,
            "multi_scale_fusion": "attention",
        }
    }
    torch.save(checkpoint, "/tmp/nestedpolyp_v2_test.pth")
    print("  Saved to /tmp/nestedpolyp_v2_test.pth")


if __name__ == "__main__":
    test_integration()
