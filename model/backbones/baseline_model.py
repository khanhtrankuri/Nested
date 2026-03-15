import torch
import torch.nn as nn

from model.backbones.unet import UNet


class BaselinePolypModel(nn.Module):
    """
    Wrapper cho baseline segmentation model.

    Trả về dict để sau này bạn dễ thống nhất với nested model:
        {
            "logits": logits
        }
    """

    def __init__(
        self,
        backbone_name: str = "unet",
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        bilinear: bool = True,
    ):
        super().__init__()

        backbone_name = backbone_name.lower()

        if backbone_name == "unet":
            self.backbone = UNet(
                in_channels=in_channels,
                out_channels=out_channels,
                bilinear=bilinear,
                base_channels=base_channels,
            )
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

    def forward(self, x: torch.Tensor):
        logits = self.backbone(x)
        return {"logits": logits}

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trả về xác suất sau sigmoid: [B, 1, H, W]
        """
        self.eval()
        logits = self.forward(x)["logits"]
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Trả về mask nhị phân: [B, 1, H, W]
        """
        probs = self.predict_proba(x)
        return (probs > threshold).float()