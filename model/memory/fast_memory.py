import torch
import torch.nn as nn


def init_fast_memory(
    batch_size: int,
    memory_dim: int,
    device,
    dtype=torch.float32,
    init_std: float = 5e-2,
):
    return torch.randn(batch_size, memory_dim, device=device, dtype=dtype) * init_std


class FastMemoryUpdater(nn.Module):
    """
    Updater cho fast memory.
    Có thể chạy:
    - kiểu cũ: feat + fast + loss
    - kiểu mới: feat + fast + slow + loss
    """

    def __init__(
        self,
        feat_dim: int,
        memory_dim: int,
        hidden_dim: int = 128,
        use_gate: bool = False,
        delta_scale: float = 1.0,
        use_slow_context: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.memory_dim = memory_dim
        self.use_gate = use_gate
        self.delta_scale = delta_scale
        self.use_slow_context = use_slow_context

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        in_dim = feat_dim + memory_dim + 1
        if self.use_slow_context:
            in_dim += memory_dim

        self.delta_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, memory_dim),
            nn.Tanh(),
        )

        if self.use_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, memory_dim),
                nn.Sigmoid(),
            )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.normal_(self.delta_mlp[-2].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.delta_mlp[-2].bias)

        if self.use_gate:
            nn.init.normal_(self.gate_mlp[-2].weight, mean=0.0, std=1e-2)
            nn.init.zeros_(self.gate_mlp[-2].bias)

    def forward(
        self,
        feat: torch.Tensor,
        memory: torch.Tensor,
        loss_scalar: torch.Tensor,
        slow_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self.pool(feat).flatten(1)

        if loss_scalar.ndim == 0:
            loss_scalar = loss_scalar.view(1, 1).repeat(memory.size(0), 1)
        elif loss_scalar.ndim == 1:
            loss_scalar = loss_scalar.view(-1, 1)

        parts = [pooled, memory]

        if self.use_slow_context:
            if slow_memory is None:
                slow_memory = torch.zeros_like(memory)
            parts.append(slow_memory)

        parts.append(loss_scalar)

        z = torch.cat(parts, dim=1)

        delta = self.delta_mlp(z)
        candidate = memory + self.delta_scale * delta

        if not self.use_gate:
            return candidate

        gate = self.gate_mlp(z)
        new_memory = gate * memory + (1.0 - gate) * candidate
        return new_memory