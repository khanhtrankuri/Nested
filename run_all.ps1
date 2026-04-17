# =============================================================================
# GlaS dataset — 10 training configurations
# =============================================================================

# [GlaS 1] Baseline swinv2_base, nested learning enabled
python train.py --dataset glas --save-root outputs/glas_nested_01/ --encoder-name swinv2_base --use-pretrained --image-size 256 256 --batch-size 4 --epochs 60 --warmup-epochs 5 --nl-apply-stages 2 3 --nl-inner-steps 1 2 3 4 --nl-inner-lr 0.0032 --enable-nested --nested-start-epoch 20 --patience 10

# [GlaS 2] swinv2_small with extended NL stages
python train.py --dataset glas --save-root outputs/glas_nested_02/ --encoder-name swinv2_small --use-pretrained --image-size 256 256 --batch-size 4 --epochs 80 --warmup-epochs 5 --nl-apply-stages 1 2 3 --nl-inner-steps 1 2 2 3 --nl-inner-lr 0.005 --nl-surprise-weight 0.08 --enable-nested --nested-start-epoch 25 --num-prototypes 12 --patience 12

# [GlaS 3] pvtv2_b2 with aggressive nested refinement
python train.py --dataset glas --save-root outputs/glas_nested_03/ --encoder-name pvtv2_b2 --use-pretrained --image-size 352 352 --batch-size 4 --epochs 100 --warmup-epochs 8 --nl-apply-stages 2 3 --nl-inner-steps 2 2 3 4 --nl-inner-lr 0.004 --enable-nested --nested-start-epoch 15 --use-ema --patience 15

# [GlaS 4] convnextv2_base with TTA and EMA
python train.py --dataset glas --save-root outputs/glas_nested_04/ --encoder-name convnextv2_base --use-pretrained --image-size 384 384 --batch-size 4 --epochs 80 --warmup-epochs 10 --nl-apply-stages 2 3 --nl-inner-steps 1 2 3 4 --nl-inner-lr 0.003 --enable-nested --nested-start-epoch 20 --use-ema --use-tta --patience 12

# [GlaS 5] swinv2_tiny fast training
python train.py --dataset glas --save-root outputs/glas_nested_05/ --encoder-name swinv2_tiny --use-pretrained --image-size 256 256 --batch-size 4 --epochs 50 --warmup-epochs 3 --nl-apply-stages 3 --nl-inner-steps 1 1 2 3 --nl-inner-lr 0.008 --enable-nested --nested-start-epoch 15 --patience 8

# [GlaS 6] pvtv2_b5 high capacity
python train.py --dataset glas --save-root outputs/glas_nested_06/ --encoder-name pvtv2_b5 --use-pretrained --image-size 352 352 --batch-size 4 --epochs 100 --warmup-epochs 8 --nl-apply-stages 1 2 3 --nl-inner-steps 2 3 3 4 --nl-inner-lr 0.002 --nl-modifier-expansion 4 --nl-surprise-weight 0.1 --enable-nested --nested-start-epoch 25 --prototype-dim 256 --num-prototypes 16 --use-ema --patience 15

# [GlaS 7] convnextv2_tiny dual-speed memory
python train.py --dataset glas --save-root outputs/glas_nested_07/ --encoder-name convnextv2_tiny --use-pretrained --image-size 256 256 --batch-size 4 --epochs 70 --warmup-epochs 5 --nl-apply-stages 2 3 --nl-inner-steps 1 2 3 4 --nl-inner-lr 0.005 --enable-nested --nested-start-epoch 20 --slow-momentum 0.004 --patience 10

# [GlaS 8] swinv2_base with stratified k-fold protocol
python train.py --dataset glas --save-root outputs/glas_nested_08/ --encoder-name swinv2_base --use-pretrained --image-size 256 256 --batch-size 4 --epochs 60 --warmup-epochs 5 --nl-apply-stages 0 1 2 3 --nl-inner-steps 1 1 2 3 --nl-inner-lr 0.003 --enable-nested --nested-start-epoch 18 --decoder-channels 192 --dropout 0.3 --nl-dropout 0.15 --patience 10

# [GlaS 9] pvtv2_b2 with val split and tuned LR
python train.py --dataset glas --save-root outputs/glas_nested_09/ --encoder-name pvtv2_b2 --use-pretrained --image-size 256 256 --batch-size 4 --epochs 80 --warmup-epochs 6 --nl-apply-stages 2 3 --nl-inner-steps 1 2 3 4 --nl-inner-lr 0.004 --encoder-lr 5e-5 --decoder-lr 2e-4 --weight-decay 5e-5 --enable-nested --nested-start-epoch 20 --glas-val-ratio 0.15 --patience 12

# [GlaS 10] convnextv2_large heavy regularization
python train.py --dataset glas --save-root outputs/glas_nested_10/ --encoder-name convnextv2_large --use-pretrained --image-size 384 384 --batch-size 4 --epochs 90 --warmup-epochs 8 --nl-apply-stages 1 2 3 --nl-inner-steps 2 2 3 4 --nl-inner-lr 0.0025 --nl-surprise-weight 0.12 --nl-modifier-expansion 4 --enable-nested --nested-start-epoch 25 --nested-max-norm 0.8 --use-ema --use-tta --patience 15


# =============================================================================
# Kvasir / Adenocarcinoma dataset — 10 training configurations
# =============================================================================

# [Kvasir 1] Baseline pvtv2_b2
python train.py --dataset kvasir --file-path datasets/Adenocarcinoma --save-root outputs/adeno_nested_01/ --encoder-name pvtv2_b2 --use-pretrained --image-size 352 352 --batch-size 4 --epochs 60 --warmup-epochs 5 --nl-apply-stages 2 3 --nl-inner-steps 1 2 3 4 --nl-inner-lr 0.0032 --enable-nested --nested-start-epoch 20 --patience 10

# [Kvasir 2] swinv2_base high-res
python train.py --dataset kvasir --file-path datasets/Adenocarcinoma --save-root outputs/adeno_nested_02/ --encoder-name swinv2_base --use-pretrained --image-size 384 384 --batch-size 4 --epochs 80 --warmup-epochs 8 --nl-apply-stages 2 3 --nl-inner-steps 1 2 3 4 --nl-inner-lr 0.003 --enable-nested --nested-start-epoch 20 --use-ema --patience 12

# [Kvasir 3] pvtv2_b5 with full-stage NL
python train.py --dataset kvasir --file-path datasets/Adenocarcinoma --save-root outputs/adeno_nested_03/ --encoder-name pvtv2_b5 --use-pretrained --image-size 352 352 --batch-size 4 --epochs 100 --warmup-epochs 10 --nl-apply-stages 0 1 2 3 --nl-inner-steps 1 2 3 4 --nl-inner-lr 0.002 --nl-surprise-weight 0.1 --nl-modifier-expansion 4 --enable-nested --nested-start-epoch 25 --prototype-dim 256 --num-prototypes 16 --use-ema --use-tta --patience 15

# [Kvasir 4] convnextv2_base balanced config
python train.py --dataset kvasir --file-path datasets/Adenocarcinoma --save-root outputs/adeno_nested_04/ --encoder-name convnextv2_base --use-pretrained --image-size 384 384 --batch-size 4 --epochs 70 --warmup-epochs 5 --nl-apply-stages 1 2 3 --nl-inner-steps 2 2 3 3 --nl-inner-lr 0.004 --enable-nested --nested-start-epoch 18 --use-ema --patience 10

# [Kvasir 5] swinv2_tiny fast baseline
python train.py --dataset kvasir --file-path datasets/Adenocarcinoma --save-root outputs/adeno_nested_05/ --encoder-name swinv2_tiny --use-pretrained --image-size 256 256 --batch-size 4 --epochs 50 --warmup-epochs 3 --nl-apply-stages 3 --nl-inner-steps 1 1 2 3 --nl-inner-lr 0.006 --enable-nested --nested-start-epoch 15 --patience 8

# [Kvasir 6] pvtv2_b2 k-fold protocol
python train.py --dataset kvasir --file-path datasets/Adenocarcinoma --save-root outputs/adeno_nested_06/ --encoder-name pvtv2_b2 --use-pretrained --image-size 352 352 --batch-size 4 --epochs 80 --warmup-epochs 5 --nl-apply-stages 2 3 --nl-inner-steps 1 2 3 4 --nl-inner-lr 0.0035 --protocol kfold --fold-index 0 --num-folds 5 --enable-nested --nested-start-epoch 20 --patience 12

# [Kvasir 7] convnextv2_tiny dual-speed memory
python train.py --dataset kvasir --file-path datasets/Adenocarcinoma --save-root outputs/adeno_nested_07/ --encoder-name convnextv2_tiny --use-pretrained --image-size 352 352 --batch-size 4 --epochs 70 --warmup-epochs 5 --nl-apply-stages 2 3 --nl-inner-steps 1 2 3 4 --nl-inner-lr 0.005 --enable-nested --nested-start-epoch 22 --slow-momentum 0.004 --patience 10

# [Kvasir 8] swinv2_small tuned LRs
python train.py --dataset kvasir --file-path datasets/Adenocarcinoma --save-root outputs/adeno_nested_08/ --encoder-name swinv2_small --use-pretrained --image-size 352 352 --batch-size 4 --epochs 80 --warmup-epochs 6 --nl-apply-stages 1 2 3 --nl-inner-steps 2 2 3 4 --nl-inner-lr 0.003 --encoder-lr 5e-5 --decoder-lr 2e-4 --weight-decay 1e-4 --enable-nested --nested-start-epoch 20 --decoder-channels 192 --patience 12

# [Kvasir 9] pvtv2_b2 frozen backbone NL
python train.py --dataset kvasir --file-path datasets/Adenocarcinoma --save-root outputs/adeno_nested_09/ --encoder-name pvtv2_b2 --use-pretrained --image-size 352 352 --batch-size 4 --epochs 60 --warmup-epochs 5 --nl-apply-stages 2 3 --nl-inner-steps 2 3 3 4 --nl-inner-lr 0.006 --nl-freeze-backbone --nl-surprise-weight 0.08 --enable-nested --nested-start-epoch 18 --patience 10

# [Kvasir 10] convnextv2_large heavy augmentation + TTA
python train.py --dataset kvasir --file-path datasets/Adenocarcinoma --save-root outputs/adeno_nested_10/ --encoder-name convnextv2_large --use-pretrained --image-size 384 384 --batch-size 4 --epochs 100 --warmup-epochs 10 --nl-apply-stages 1 2 3 --nl-inner-steps 2 3 3 4 --nl-inner-lr 0.0025 --nl-modifier-expansion 4 --nl-dropout 0.15 --dropout 0.3 --enable-nested --nested-start-epoch 25 --nested-max-norm 0.8 --use-ema --use-tta --tta-scales 1.0 0.75 1.25 1.5 --patience 15
