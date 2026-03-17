# NBM-PolypNet-Hybrid

`NBM-PolypNet-Hybrid` is a practical polyp segmentation model that keeps the base network strong on its own and uses memory only as a small residual refinement at selected decoder stages.

## Architecture

- High-resolution detail stem keeps shallow texture and boundary cues at full resolution.
- Hybrid hierarchical encoder uses CNN stages at `1/2` and `1/4`, then lightweight transformer-style stages at `1/8` and `1/16`.
- Hybrid bottleneck neck combines `PPM` context and `ASPP-lite` context before decoding.
- Decoder follows a U-Net top-down path with partial UNet++-style nested skip bridges near the upper decoder.
- Edge guidance branch produces both edge logits and multi-scale edge features that gate decoder refinement.
- Reverse-attention head predicts a coarse mask first, then adds a restrained residual refinement.
- Safe prototype memory is only attached at `1/8` and `1/4` decoder stages and only adds a bounded residual correction.

## Why This Memory Design Is Safer

- Memory never replaces the backbone prediction. It only contributes a low-amplitude residual.
- Memory is limited to two decoder stages, not injected everywhere.
- Validation and test freeze memory by default, so there is no label leakage or test-time memory drift.
- Fast memory updates use confidence-aware gating and norm clipping.
- Slow memory updates happen per task with EMA-style blending and max-norm clipping.
- `--skip-memory-if-hurts` falls back to the base path when a memory update makes the batch worse.

## File Layout

- `model/modules_hybrid.py`: reusable encoder, neck, decoder, edge, reverse-attention, and memory modules.
- `model/nbm_polyp_hybrid.py`: assembled `NBMPolypNetHybrid` model.
- `loss/hybrid_losses.py`: `SoftDiceLoss`, `FocalTverskyLoss`, `EdgeLoss`, and `HybridPolypLoss`.
- `engine/train_one_epoch_hybrid.py`: warmup, task, and joint training helpers.
- `engine/validate_hybrid.py`: validation with threshold sweep and frozen-memory eval.
- `engine/test_hybrid.py`: test-time evaluation with optional TTA and mask export.
- `train_nbm_polyp_hybrid.py`: full warmup, curriculum, and joint finetune training entrypoint.

## Train

```bash
python train_nbm_polyp_hybrid.py \
  --file-path datasets/Kvasir \
  --save-root outputs/kvasir_nbm_polyp_hybrid \
  --image-size 352 352 \
  --batch-size 8 \
  --num-workers 4 \
  --num-tasks 4 \
  --warmup-epochs 8 \
  --warmup-lr 1e-4 \
  --task-epochs 3,4,5,6 \
  --task-lrs 8e-5,6e-5,5e-5,4e-5 \
  --joint-finetune-epochs 6 \
  --joint-finetune-lr 2e-5 \
  --memory-start-task 2 \
  --memory-blend 0.10 \
  --slow-memory-max-norm 1.0 \
  --num-prototypes 6 \
  --memory-dim 64 \
  --base-channels 32 \
  --train-augmentation \
  --freeze-eval-memory \
  --use-tta \
  --skip-memory-if-hurts
```

## Evaluate Existing Checkpoint

Run the same script with the same output directory after training. It saves:

- `best_model.pth`
- `history.json`
- `test_metrics.json`
- `predictions/`

The validation path performs threshold sweep over `0.35-0.65`, and the test path can use original, horizontal flip, vertical flip, and double-flip TTA.
