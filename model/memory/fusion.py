import torch
import torch.nn as nn


class MemoryFiLMFusion(nn.Module):
    """
    Dual-memory FiLM fusion:
    - fast memory: trạng thái thích nghi ngắn hạn
    - slow memory: prior liên-task

    Nếu slow_memory=None thì vẫn chạy như bản cũ.
    """

    def __init__(self, feat_dim: int, memory_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feat_dim = feat_dim
        self.memory_dim = memory_dim

        # context:
        # [fast, slow, fast-slow, fast*slow]
        self.context_dim = memory_dim * 4

        self.context_mlp = nn.Sequential(
            nn.Linear(self.context_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.to_gamma = nn.Linear(hidden_dim, feat_dim)
        self.to_beta = nn.Linear(hidden_dim, feat_dim)

        self.refine = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # lớp cuối init nhỏ để tránh phá bottleneck quá mạnh lúc đầu
        nn.init.normal_(self.to_gamma.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.to_gamma.bias)

        nn.init.normal_(self.to_beta.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.to_beta.bias)

    def _build_context(
        self,
        fast_memory: torch.Tensor,
        slow_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if slow_memory is None:
            slow_memory = torch.zeros_like(fast_memory)

        context = torch.cat(
            [
                fast_memory,
                slow_memory,
                fast_memory - slow_memory,
                fast_memory * slow_memory,
            ],
            dim=1,
        )
        return context

    def forward(
        self,
        feat: torch.Tensor,
        fast_memory: torch.Tensor,
        slow_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self._build_context(fast_memory, slow_memory)
        h = self.context_mlp(context)

        gamma = self.to_gamma(h).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(h).unsqueeze(-1).unsqueeze(-1)

        fused = feat * (1.0 + gamma) + beta
        fused = self.refine(fused)
        return fused