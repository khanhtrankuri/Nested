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
- **Paper-inspired nested memory option**: fast/slow prototype memory with learned gates

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
 CUDA_VISIBLE_DEVICES=7 python train_baseline_clean.py --file-path datasets/Adenocarcinoma --save-root outputs/Adenocarcinoma --batch-size 16 --num-workers 8 --epochs 100 --encoder-name pvtv2_b5 --use-ema --use-tta --enable-nested --nested-start-epoch 20 --nested-memory-mode fast_slow --nested-dim 256 --pretrained-cache-dir ./pvt_v2_b5_weights
```

## Why this baseline is safer
The current repo baseline is still a plain U-Net wrapper, uses a BCE+Dice objective, and the default training script picks checkpoints by validation Dice. This patch moves to a stronger encoder-decoder backbone, stronger augmentation, and IoU-first checkpointing.
