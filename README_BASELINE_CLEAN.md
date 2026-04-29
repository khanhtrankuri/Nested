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

## Recommended first run with glas dataset
```bash
 python train_baseline_clean.py   --dataset glas   --image-size 512 512   --save-root outputs/glas/swin_v3   --batch-size 2 --num-workers 4   --epochs 80   --encoder-name swinv2_base   --use-pretrained   --encoder-lr 1e-5 --decoder-lr 1e-4   --use-ema --ema-decay 0.99   --use-tta --tta-scales 1.0   --enable-nested --nested-start-epoch 10   --nested-memory-mode fast_slow   --glas-val-ratio 0.15   --patience 10
```

## Recommended more reliable run
```bash
 CUDA_VISIBLE_DEVICES=6 python train_baseline_clean.py     --file-path datasets/Adenocarcinoma     --save-root outputs/Adenocarcinoma/swin_v3     --encoder-name swinv2_base     --use-pretrained     --image-size 256 256     --batch-size 32     --epochs 80     --warmup-epochs 8     --patience 15     --enable-nested     --nested-start-epoch 12     --nested-dim 128     --nested-prototypes 8     --nested-residual-scale 0.08     --nested-memory-mode fast_slow     --nested-memory-hidden 128     --nested-slow-momentum-scale 0.35     --use-ema --ema-decay 0.9995     --use-tta --tta-scales 1.0      --dropout 0.1     --small-polyp-sampling-power 0.35     --thresholds 0.40 0.45 0.50 0.55 
```



## Why this baseline is safer
The current repo baseline is still a plain U-Net wrapper, uses a BCE+Dice objective, and the default training script picks checkpoints by validation Dice. This patch moves to a stronger encoder-decoder backbone, stronger augmentation, and IoU-first checkpointing.


python train_baseline_clean.py --file-path datasets/Adenocarcinoma --save-root outputs/Adenocarcinoma/swin_new --batch-size 16 --num-workers 4 --epochs 100 --encoder-name swinv2_base --use-pretrained --image-size 256 256 --use-ema --use-tta --tta-scales 1.0 --enable-nested --nested-start-epoch 20 --nested-memory-mode fast_slow 

python train.py --dataset kvasir --file-path datasets/Adenocarcinoma --encoder-name  swinv2_base --use-pretrained --batch-size 4 --epochs 60 --enable-nested


