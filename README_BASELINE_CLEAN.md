# Clean Strong Baseline for Nested repo

This patch gives you a **simple but stronger baseline** than the current plain U-Net in the repo.

## What changed
- **No memory**
- **No curriculum / task training**
- **No complex refine path**
- **Stronger encoder**: pure-PyTorch ConvNeXt-like hierarchical encoder
- **FPN decoder** with auxiliary head
- **Stronger loss**: BCE + Lovasz + Focal Tversky + Dice + aux
- **Stronger augmentation** than the current repo baseline
- **Strict split or 5-fold CV**
- **EMA + threshold sweep + optional TTA**
- **Model selection by IoU first**

## Files
- `data/load_data_clean.py`
- `model/backbones/strong_baseline.py`
- `loss/strong_baseline_loss.py`
- `engine/train_eval_clean.py`
- `train_baseline_clean.py`

## Recommended first run
```bash
python train_baseline_clean.py \
  --file-path datasets/Kvasir \
  --save-root outputs/kvasir_clean_baseline \
  --image-size 384 384 \
  --batch-size 8 \
  --epochs 60 \
  --protocol strict \
  --use-ema \
  --use-tta
```

## Recommended more reliable run
```bash
python train_baseline_clean.py \
  --file-path datasets/Kvasir \
  --save-root outputs/kvasir_clean_baseline_fold0 \
  --image-size 384 384 \
  --batch-size 8 \
  --epochs 80 \
  --protocol kfold \
  --fold-index 0 \
  --num-folds 5 \
  --use-ema \
  --use-tta
```

## Why this baseline is safer
The current repo baseline is still a plain U-Net wrapper, uses a BCE+Dice objective, and the default training script picks checkpoints by validation Dice. This patch moves to a stronger encoder-decoder backbone, stronger augmentation, and IoU-first checkpointing.
