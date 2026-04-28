#!/usr/bin/env python3
"""
Demo script for Advanced PolyMemnet v2.0

Showcases all 6 enhancements with a simple synthetic data test.
"""

import torch
from model.advanced_polymemnet import AdvancedPolyMemnet


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model with all enhancements enabled
    model = AdvancedPolyMemnet(
        encoder_name="tiny_convnext",
        decoder_channels=128,
        enable_nested=True,
        # Enhancement 1: Cross-Stage Modulation
        use_cross_stage_encoder=True,
        # Enhancement 2: Adaptive CMS
        use_adaptive_cms=True,
        adaptive_max_steps=8,
        # Enhancement 3: Meta Optimizer (via adaptive blocks)
        # Enhancement 4: Hierarchical Prototypes
        use_hierarchical_prototypes=True,
        hierarchical_levels=3,
        prototypes_per_level=[4, 4, 4],  # total 12 prototypes across levels
        cms_levels=[0, 1, 2, 3],
        # Enhancement 5: MC Dropout
        use_mc_dropout=True,
        mc_dropout_p=0.1,
        num_mc_samples=5,
        # Enhancement 6: Enhanced Loss
        use_enhanced_loss=True,
        prototype_contrastive_weight=0.05,
        inner_consistency_weight=0.02,
        gate_sparsity_weight=0.01,
        memory_quality_weight=0.02,
    ).to(device)

    print("\n" + "="*70)
    print("Advanced PolyMemnet Architecture")
    print("="*70)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")

    # Print enabled features
    print("\nEnabled features:")
    print(f"  Cross-stage modulation: {model.use_cross_stage_encoder}")
    print(f"  Adaptive CMS: {model.use_adaptive_cms}")
    print(f"  Advanced decoder: {model.use_advanced_decoder}")
    print(f"  Hierarchical prototypes: {model.use_hierarchical_prototypes}")
    print(f"  MC Dropout: {model.use_mc_dropout}")
    print(f"  Enhanced loss: {model.use_enhanced_loss}")

    # Test forward pass
    print("\n" + "="*70)
    print("Testing forward pass...")
    print("="*70)
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 384, 384).to(device)
        out = model(x, use_nested=True, return_uncertainty=True, num_mc_samples=5)

        print(f"\nInput shape: {x.shape}")
        print(f"Logits shape: {out['logits'].shape}")
        print(f"Coarse logits shape: {out['coarse_logits'].shape}")
        print(f"Aux logits shape: {out['aux_logits'].shape}")

        if 'uncertainty' in out:
            print(f"Uncertainty shape: {out['uncertainty'].shape}")

        # Nested info
        nested_info = out.get('nested_info', {})
        print(f"\nNested info keys: {list(nested_info.keys())}")
        if 'cross_stage' in nested_info:
            print("  Cross-stage modulation active")
        if 'cms_gate_values' in nested_info:
            print(f"  CMS gate values: {nested_info['cms_gate_values']}")

    # Test loss computation
    print("\n" + "="*70)
    print("Testing loss computation...")
    print("="*70)
    model.train()
    with torch.no_grad():
        x = torch.randn(2, 3, 384, 384).to(device)
        y = torch.randint(0, 2, (2, 1, 384, 384)).float().to(device)
        out = model(x, use_nested=True)

        if hasattr(model, 'compute_loss'):
            total_loss, loss_dict = model.compute_loss(out, y)
            print(f"\nTotal loss: {total_loss.item():.4f}")
            print("Loss components:")
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}")
        else:
            print("Using standard StrongBaselineLoss")
            from loss.strong_baseline_loss import StrongBaselineLoss
            criterion = StrongBaselineLoss()
            loss = criterion(out, y)
            print(f"Loss: {loss.item():.4f}")

    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
