"""
Boundary Refinement Module — Focus on edge pixels and high-frequency details.

Paper insight: "Boundary quality is critical for medical segmentation. Standard
decoders smooth out fine details, especially for small polyps."

Design:
1. Edge detection via Sobel filters (fixed, non-trainable)
2. High-frequency feature extraction (Laplacian)
3. Learnable edge attention to weight refinement strength
4. Gated residual connection to avoid over-smoothing

Integration: Add to decoder fused features before seg_head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SobelEdgeDetector(nn.Module):
    """Fixed Sobel filters for edge detection (no gradients)."""

    def __init__(self):
        super().__init__()
        # Register buffers (not parameters) — fixed weights
        self.register_buffer(
            "sobel_x",
            torch.tensor([[[[-1.0, 0.0, 1.0],
                           [-2.0, 0.0, 2.0],
                           [-1.0, 0.0, 1.0]]]], dtype=torch.float32)
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[[[-1.0, -2.0, -1.0],
                           [0.0, 0.0, 0.0],
                           [1.0, 2.0, 1.0]]]], dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input feature map

        Returns:
            edge_mag: (B, 1, H, W) edge magnitude (normalized to [0, 1])
        """
        B, C, H, W = x.shape

        # Apply Sobel per channel, then average
        edge_x_per_channel = []
        edge_y_per_channel = []
        for c in range(C):
            ch = x[:, c:c+1, :, :]
            gx = F.conv2d(ch, self.sobel_x, padding=1)
            gy = F.conv2d(ch, self.sobel_y, padding=1)
            edge_x_per_channel.append(gx)
            edge_y_per_channel.append(gy)

        edge_x = torch.stack(edge_x_per_channel, dim=1).mean(dim=1, keepdim=False)
        edge_y = torch.stack(edge_y_per_channel, dim=1).mean(dim=1, keepdim=False)

        # Magnitude
        edge_mag = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)

        # Normalize per image to [0, 1]
        # Use amin/amax instead of min/max with list of dims
        edge_min = edge_mag.amin(dim=[1, 2, 3], keepdim=True)
        edge_max = edge_mag.amax(dim=[1, 2, 3], keepdim=True)
        edge_norm = (edge_mag - edge_min) / (edge_max - edge_min + 1e-8)

        return edge_norm


class HighFrequencyExtractor(nn.Module):
    """Extract high-frequency components using Laplacian."""

    def __init__(self, channels: int):
        super().__init__()
        # Laplacian kernel (3x3)
        laplacian_kernel = torch.tensor([[[[0.0, 1.0, 0.0],
                                           [1.0, -4.0, 1.0],
                                           [0.0, 1.0, 0.0]]]], dtype=torch.float32)
        self.register_buffer("laplacian", laplacian_kernel)

        # Projection to process high-freq features
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, max(1, (channels // 2) // 4)), channels // 2),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features

        Returns:
            hf_features: (B, C, H, W) high-frequency enhanced features
        """
        B, C, H, W = x.shape

        # Apply Laplacian per channel
        hf_list = []
        for c in range(C):
            ch = x[:, c:c+1, :, :]
            hf_ch = F.conv2d(ch, self.laplacian, padding=1)
            hf_list.append(hf_ch)
        hf = torch.stack(hf_list, dim=1).squeeze(2)  # (B, C, H, W)

        # Process high-frequency features
        hf_processed = self.proj(hf)

        return hf_processed


class BoundaryAwareGate(nn.Module):
    """Learn to modulate refinement strength based on boundary confidence."""

    def __init__(self, channels: int):
        super().__init__()
        # Input: concatenated x (C) and edge_map (1) → total channels+1
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels + 1, channels // 4, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, max(1, (channels // 4) // 4)), channels // 4),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1),  # This conv has bias
            nn.Sigmoid()
        )
        # Initialize bias low → conservative gating initially
        self.gate_net[-2].bias.data.fill_(-2.0)  # Second-to-last layer is the conv with bias

    def forward(self, x: torch.Tensor, edge_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) fused features
            edge_map: (B, 1, H, W) edge magnitude

        Returns:
            gate: (B, 1, H, W) per-pixel gate in [0, 1]
        """
        # Concatenate edge information
        combined = torch.cat([x, edge_map], dim=1)
        gate = self.gate_net(combined)
        return gate


class BoundaryRefinementModule(nn.Module):
    """
    Main boundary refinement module.

    Architecture:
        fused_features → [EdgeDetector + HFExtractor] → refine_features
                                                      ↑
                                              EdgeAwareGate controls amount

    Formula:
        edge = Sobel(fused)
        hf = Laplacian(fused)
        gate = sigmoid(Conv([fused, edge]))
        refined = fused + gate * (edge_enhance + hf_processed)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        self.edge_detector = SobelEdgeDetector()
        self.hf_extractor = HighFrequencyExtractor(channels)
        self.boundary_gate = BoundaryAwareGate(channels)

        # Optional: lightweight residual block for refinement
        self.refine_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, max(1, channels // 4)), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.1))  # conservative init

    def forward(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused: (B, C, H, W) fused features from decoder

        Returns:
            dict with:
                'refined': (B, C, H, W) boundary-enhanced features
                'edge_map': (B, 1, H, W) edge magnitude for logging
                'gate_mean': scalar mean gate value
        """
        B, C, H, W = fused.shape
        assert C == self.channels, f"Expected {self.channels} channels, got {C}"

        # 1. Edge detection
        edge_map = self.edge_detector(fused)  # (B, 1, H, W)

        # 2. High-frequency extraction
        hf_features = self.hf_extractor(fused)  # (B, C, H, W)

        # 3. Compute gate (where to apply refinement)
        gate = self.boundary_gate(fused, edge_map)  # (B, 1, H, W)

        # 4. Combine high-frequency + edge-guided
        # Edge map broadcast to C channels
        edge_enhanced = edge_map * hf_features

        # 5. Apply refine block
        refine_feat = self.refine_block(fused + edge_enhanced)

        # 6. Gated residual
        refined = fused + self.residual_scale * gate * refine_feat

        return {
            'refined': refined,
            'edge_map': edge_map.detach(),
            'gate_mean': gate.mean().detach(),
        }


def _smoke_test():
    """Quick test for BoundaryRefinementModule."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, C, H, W = 2, 128, 96, 96

    print("=" * 60)
    print("BoundaryRefinementModule Smoke Test")
    print("=" * 60)

    module = BoundaryRefinementModule(channels=C).to(device)
    fused = torch.randn(B, C, H, W, device=device, requires_grad=True)

    print("\n[1] Forward pass")
    out = module(fused)
    refined = out['refined']
    edge_map = out['edge_map']
    gate_mean = out['gate_mean']

    print(f"  Input: {tuple(fused.shape)}")
    print(f"  Refined: {tuple(refined.shape)}")
    print(f"  Edge map: {tuple(edge_map.shape)}")
    print(f"  Gate mean: {gate_mean.item():.4f}")
    assert refined.shape == fused.shape
    assert edge_map.shape == (B, 1, H, W)
    print("  PASS")

    print("\n[2] Gradient flow")
    loss = refined.sum() + 0.1 * gate_mean
    loss.backward()
    assert fused.grad is not None
    assert fused.grad.abs().sum() > 0
    # Check module parameters have grads
    for name, param in module.named_parameters():
        assert param.grad is not None, f"No grad for {name}"
    print(f"  All parameters have gradients")
    print("  PASS")

    print("\n[3] Edge map statistics")
    edge_min = edge_map.min().item()
    edge_max = edge_map.max().item()
    edge_mean = edge_map.mean().item()
    print(f"  Edge map range: [{edge_min:.3f}, {edge_max:.3f}], mean={edge_mean:.3f}")
    assert 0.0 <= edge_min and edge_max <= 1.0, "Edge map not normalized"
    print("  PASS")

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    _smoke_test()
