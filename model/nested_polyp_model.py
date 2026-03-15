import torch
import torch.nn as nn

from model.backbones.unet_nl import UNet_NL
from model.memory.fast_memory import FastMemoryUpdater, init_fast_memory
from model.memory.fusion import MemoryFiLMFusion


class NestedLitePolypModel(nn.Module):
    """
    Nested-lite với:
    - fast memory: update theo batch
    - slow memory: EMA theo task
    - bottleneck dùng đồng thời fast + slow memory
    """

    def __init__(
        self,
        backbone_name: str = "unet_nl",
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        bilinear: bool = True,
        memory_dim: int = 64,
        updater_hidden_dim: int = 128,
        use_gate: bool = True,
        fast_init_std: float = 5e-2,
        slow_init_std: float = 1e-2,
    ):
        super().__init__()

        backbone_name = backbone_name.lower()
        if backbone_name != "unet_nl":
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        self.backbone = UNet_NL(
            in_channels=in_channels,
            out_channels=out_channels,
            bilinear=bilinear,
            base_channels=base_channels,
        )

        self.memory_dim = memory_dim
        self.fast_init_std = fast_init_std
        self.slow_init_std = slow_init_std

        feat_dim = self.backbone.bottleneck_channels

        self.memory_fusion = MemoryFiLMFusion(
            feat_dim=feat_dim,
            memory_dim=memory_dim,
            hidden_dim=updater_hidden_dim,
        )

        self.memory_updater = FastMemoryUpdater(
            feat_dim=feat_dim,
            memory_dim=memory_dim,
            hidden_dim=updater_hidden_dim,
            use_gate=use_gate,
            use_slow_context=True,
        )

        self.register_buffer("slow_memory", torch.zeros(memory_dim))
        self.reset_slow_memory()

    @torch.no_grad()
    def reset_slow_memory(self):
        self.slow_memory.normal_(mean=0.0, std=self.slow_init_std)

    def get_slow_memory(self, batch_size=None, device=None, dtype=torch.float32):
        slow = self.slow_memory
        if device is not None or dtype is not None:
            slow = slow.to(device=device, dtype=dtype)

        if batch_size is None:
            return slow

        return slow.unsqueeze(0).repeat(batch_size, 1)

    def init_memory(
        self,
        batch_size: int,
        device,
        dtype=torch.float32,
        from_slow: bool = True,
        noise_std: float = 1e-3,
        slow_scale: float = 0.2,
    ):
        """
        Khởi tạo fast memory từ slow prior hoặc random nhỏ.
        """
        if from_slow:
            memory = self.get_slow_memory(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            memory = slow_scale * memory
            if noise_std > 0:
                memory = memory + noise_std * torch.randn_like(memory)
            return memory

        return init_fast_memory(
            batch_size=batch_size,
            memory_dim=self.memory_dim,
            device=device,
            dtype=dtype,
            init_std=self.fast_init_std,
        )

    @torch.no_grad()
    def update_slow_memory(
        self,
        task_memory_summary: torch.Tensor,
        momentum: float = 0.05,
        max_norm: float = 1.0,
    ):
        if task_memory_summary.ndim == 2:
            task_memory_summary = task_memory_summary.mean(dim=0)

        task_memory_summary = task_memory_summary.detach().to(
            device=self.slow_memory.device,
            dtype=self.slow_memory.dtype,
        )

        self.slow_memory.mul_(1.0 - momentum).add_(momentum * task_memory_summary)

        norm = self.slow_memory.norm(p=2)
        if norm > max_norm:
            self.slow_memory.mul_(max_norm / (norm + 1e-6))

    def forward(self, x: torch.Tensor, memory: torch.Tensor):
        """
        memory ở đây là fast memory.
        slow memory sẽ được lấy tự động từ buffer.
        """
        features = self.backbone.encode(x)
        feat = features["x5"]

        batch_size = x.size(0)
        slow_batch = self.get_slow_memory(
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )

        fused_feat = self.memory_fusion(
            feat=feat,
            fast_memory=memory,
            slow_memory=slow_batch,
        )

        logits = self.backbone.decode(features, bottleneck=fused_feat)

        return {
            "logits": logits,
            "feat": feat,
            "fused_feat": fused_feat,
            "fast_memory": memory,
            "slow_memory": slow_batch,
        }

    def compute_updated_memory(
        self,
        feat: torch.Tensor,
        memory: torch.Tensor,
        loss_scalar: torch.Tensor,
    ) -> torch.Tensor:
        slow_batch = self.get_slow_memory(
            batch_size=memory.size(0),
            device=memory.device,
            dtype=memory.dtype,
        )
        return self.memory_updater(
            feat=feat,
            memory=memory,
            loss_scalar=loss_scalar,
            slow_memory=slow_batch,
        )

    @torch.no_grad()
    def update_memory(
        self,
        feat: torch.Tensor,
        memory: torch.Tensor,
        loss_scalar: torch.Tensor,
    ) -> torch.Tensor:
        slow_batch = self.get_slow_memory(
            batch_size=memory.size(0),
            device=memory.device,
            dtype=memory.dtype,
        )
        return self.memory_updater(
            feat=feat,
            memory=memory,
            loss_scalar=loss_scalar,
            slow_memory=slow_batch,
        )

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x, memory)["logits"]
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_mask(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        probs = self.predict_proba(x, memory)
        return (probs > threshold).float()