import torch


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-7):
    return numerator / (denominator + eps)


@torch.no_grad()
def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    dims = (1, 2, 3)
    tp = (preds * targets).sum(dim=dims)
    fp = (preds * (1.0 - targets)).sum(dim=dims)
    fn = ((1.0 - preds) * targets).sum(dim=dims)

    intersection = tp
    pred_area = preds.sum(dim=dims)
    target_area = targets.sum(dim=dims)
    union = pred_area + target_area - intersection

    dice = _safe_div(2.0 * intersection, pred_area + target_area, eps)
    iou = _safe_div(intersection, union, eps)
    precision = _safe_div(tp, tp + fp, eps)
    recall = _safe_div(tp, tp + fn, eps)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }