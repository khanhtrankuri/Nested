# Tích hợp CMSSelfModifyingEncoder vào train.py

Tài liệu này mô tả cách thay thế `SelfModifyingEncoder` cũ bằng
`CMSSelfModifyingEncoder` mới để có hierarchy cập nhật đúng nguyên lý
Continuum Memory System.

## Tóm tắt khác biệt

| Điểm | Bản cũ | Bản mới (CMS) |
|---|---|---|
| Stage được NL-hoá | `apply_stages=[2,3]` (all-or-nothing) | Per-stage mode: `none` / `light` / `full` |
| Inner LR | `base_lr * 0.5^stage_idx` | Per-stage rõ ràng trong config |
| Inner optimizer | Plain SGD | SGD + momentum (persistent cho deep) |
| Surprise loss | Luôn là recon + consistency | Per-stage: `consistency` / `full` |
| Backbone LR | 1 giá trị duy nhất | LLRD theo depth |
| Progressive unfreeze | Không có | `set_epoch()` mở khóa dần |
| Stage nông | Bỏ qua | Có LoRA-like adaptor (rank 4) |

## Bước 1 — Copy file

Đặt `self_modifying_encoder_cms.py` vào `model/backbones/`:

```
model/backbones/
├── strong_baseline.py
├── cms_decoder.py
├── self_modifying_encoder.py           # cũ, giữ lại làm fallback
└── self_modifying_encoder_cms.py       # mới
```

## Bước 2 — Sửa `NestedPolypModel.__init__` trong `train.py`

Thay đoạn tạo encoder:

```python
# CŨ
from model.backbones.self_modifying_encoder import SelfModifyingEncoder
...
self.encoder = SelfModifyingEncoder(
    backbone=backbone,
    feature_channels=self.backbone_channels,
    inner_steps_schedule=nl_inner_steps,
    inner_lr=nl_inner_lr,
    apply_stages=self.nl_apply_stages,
    freeze_backbone=nl_freeze_backbone,
    modifier_expansion=nl_modifier_expansion,
    dropout=nl_dropout,
)
```

Bằng:

```python
# MỚI
from model.backbones.self_modifying_encoder_cms import (
    CMSSelfModifyingEncoder, DEFAULT_STAGE_CONFIG,
)

# Cho phép override từ CLI nếu muốn, nhưng thường default đã đủ tốt
stage_configs = DEFAULT_STAGE_CONFIG
# Nếu bạn muốn tuỳ biến, ví dụ tắt hẳn stage c3:
# stage_configs = [
#     {"mode": "none"},                                    # c2
#     {"mode": "none"},                                    # c3 (tắt)
#     {"mode": "full", "inner_steps": 2, "inner_lr": 5e-3, ...},   # c4
#     {"mode": "full", "inner_steps": 4, "inner_lr": 2e-2, ...},   # c5
# ]

self.encoder = CMSSelfModifyingEncoder(
    backbone=backbone,
    feature_channels=self.backbone_channels,
    stage_configs=stage_configs,
    backbone_lr_decay=0.5,
    unfreeze_schedule={
        "deep": 0,       # c4, c5 mở khóa ngay
        "mid": nl_mid_unfreeze_epoch,      # ví dụ 8
        "shallow": nl_shallow_unfreeze_epoch,  # ví dụ 16
    },
)
```

Lưu ý: bỏ các tham số cũ `nl_inner_steps`, `nl_inner_lr`,
`nl_apply_stages`, `nl_modifier_expansion` khỏi constructor — chúng bây
giờ nằm trong `stage_configs`.

## Bước 3 — Sửa `build_parser()`

Thêm các flag mới, bỏ các flag cũ không còn dùng:

```python
# BỎ:
# parser.add_argument("--nl-inner-steps", type=int, nargs=4, default=[1, 2, 3, 4])
# parser.add_argument("--nl-inner-lr", type=float, default=0.01)
# parser.add_argument("--nl-apply-stages", type=int, nargs="+", default=[2, 3])
# parser.add_argument("--nl-modifier-expansion", type=int, default=2)
# parser.add_argument("--nl-freeze-backbone", action="store_true")

# THÊM:
parser.add_argument("--nl-stage-config-preset",
                    choices=["default", "conservative", "aggressive", "custom"],
                    default="default",
                    help="Preset cho stage_configs. 'custom' dùng DEFAULT_STAGE_CONFIG "
                         "kèm override từ các flag bên dưới.")
parser.add_argument("--nl-c4-inner-steps", type=int, default=2)
parser.add_argument("--nl-c5-inner-steps", type=int, default=4)
parser.add_argument("--nl-c4-inner-lr", type=float, default=5e-3)
parser.add_argument("--nl-c5-inner-lr", type=float, default=2e-2)
parser.add_argument("--backbone-lr-decay", type=float, default=0.5)
parser.add_argument("--unfreeze-deep-epoch", type=int, default=0)
parser.add_argument("--unfreeze-mid-epoch", type=int, default=8)
parser.add_argument("--unfreeze-shallow-epoch", type=int, default=16)
parser.add_argument("--adaptor-lr", type=float, default=3e-4)
```

Thay preset thành config tương ứng:

```python
def _build_stage_configs(args):
    from model.backbones.self_modifying_encoder_cms import DEFAULT_STAGE_CONFIG
    import copy

    if args.nl_stage_config_preset == "default":
        return DEFAULT_STAGE_CONFIG

    if args.nl_stage_config_preset == "conservative":
        # Chỉ inner loop ở c5, c4 dùng light adaptor
        return [
            {"mode": "none"},
            {"mode": "none"},
            {"mode": "light", "lora_rank": 8, "residual_init": 0.05},
            {"mode": "full", "inner_steps": 2, "inner_lr": 1e-2,
             "inner_momentum": 0.9, "modifier_expansion": 2,
             "surprise_type": "full", "persist_momentum": True,
             "residual_init": 0.05},
        ]

    if args.nl_stage_config_preset == "aggressive":
        # Inner loop cả c3, c4, c5 với K tăng dần
        return [
            {"mode": "light", "lora_rank": 4, "residual_init": 0.03},
            {"mode": "full", "inner_steps": 1, "inner_lr": 2e-3,
             "inner_momentum": 0.85, "modifier_expansion": 1,
             "surprise_type": "consistency", "persist_momentum": False,
             "residual_init": 0.03},
            {"mode": "full", "inner_steps": 3, "inner_lr": 1e-2,
             "inner_momentum": 0.9, "modifier_expansion": 2,
             "surprise_type": "full", "persist_momentum": True,
             "residual_init": 0.05},
            {"mode": "full", "inner_steps": 5, "inner_lr": 3e-2,
             "inner_momentum": 0.95, "modifier_expansion": 4,
             "surprise_type": "full", "persist_momentum": True,
             "residual_init": 0.08},
        ]

    # custom: override từ CLI
    cfg = copy.deepcopy(DEFAULT_STAGE_CONFIG)
    cfg[2]["inner_steps"] = args.nl_c4_inner_steps
    cfg[2]["inner_lr"] = args.nl_c4_inner_lr
    cfg[3]["inner_steps"] = args.nl_c5_inner_steps
    cfg[3]["inner_lr"] = args.nl_c5_inner_lr
    return cfg
```

## Bước 4 — Thay thế optimizer (QUAN TRỌNG)

Bản cũ:

```python
param_groups = model.get_parameter_groups()
encoder_params = param_groups["encoder"]
decoder_params = param_groups["decoder"]
optimizer = torch.optim.AdamW(
    [
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": decoder_params, "lr": args.decoder_lr},
    ],
    weight_decay=args.weight_decay,
)
```

Bản mới — dùng `build_param_groups()` từ encoder để có LLRD, thêm
group riêng cho decoder:

```python
encoder_groups = model.encoder.build_param_groups(
    base_backbone_lr=args.encoder_lr,
    adaptor_lr=args.adaptor_lr,
    modifier_inner_lr_scale=0.5,
    weight_decay=args.weight_decay,
)

# Decoder: mọi tham số ngoài encoder
encoder_param_ids = set()
for g in encoder_groups:
    for p in g["params"]:
        encoder_param_ids.add(id(p))
decoder_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]

all_groups = encoder_groups + [{
    "params": decoder_params,
    "lr": args.decoder_lr,
    "weight_decay": args.weight_decay,
    "name": "decoder",
}]

optimizer = torch.optim.AdamW(all_groups)

# In ra để kiểm tra
print("[Optimizer] Parameter groups:")
for g in all_groups:
    n = sum(p.numel() for p in g["params"])
    print(f"  {g.get('name', '?'):<30} lr={g['lr']:.2e}  wd={g['weight_decay']:.1e}  params={n:,}")
```

## Bước 5 — Gọi `set_epoch()` mỗi epoch

Trong vòng lặp training, **trước** khi gọi `train_one_epoch_clean`:

```python
for epoch in range(1, args.epochs + 1):
    # MỚI: cập nhật trạng thái freeze theo schedule
    model.encoder.set_epoch(epoch)

    # Lưu ý: nếu optimizer đã tạo với requires_grad=False cho một số
    # params, sau khi set_epoch unfreeze, optimizer sẽ KHÔNG tự động
    # thấy param mới. Cách an toàn: khi set_epoch thay đổi trạng thái
    # unfreeze (ví dụ từ "mid" lên "shallow"), ta REBUILD optimizer.
    #
    # Đơn giản nhất: rebuild đầu mỗi epoch nếu biết có thay đổi.
    if epoch == args.unfreeze_mid_epoch or epoch == args.unfreeze_shallow_epoch:
        old_state = optimizer.state_dict()
        encoder_groups = model.encoder.build_param_groups(
            base_backbone_lr=args.encoder_lr,
            adaptor_lr=args.adaptor_lr,
            modifier_inner_lr_scale=0.5,
            weight_decay=args.weight_decay,
        )
        encoder_param_ids = set()
        for g in encoder_groups:
            for p in g["params"]:
                encoder_param_ids.add(id(p))
        decoder_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
        all_groups = encoder_groups + [{
            "params": decoder_params,
            "lr": args.decoder_lr,
            "weight_decay": args.weight_decay,
            "name": "decoder",
        }]
        optimizer = torch.optim.AdamW(all_groups)
        # (Không load lại old_state vì params layout khác; chấp nhận reset momentum)

        # Rebuild scheduler tương ứng
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.25,
            total_iters=max(args.warmup_epochs, 1))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - epoch, 1), eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[args.warmup_epochs])

        print(f"[Epoch {epoch}] Rebuilt optimizer after unfreeze. New groups:")
        for g in all_groups:
            n = sum(p.numel() for p in g["params"])
            print(f"  {g.get('name', '?'):<30} lr={g['lr']:.2e}  params={n:,}")

    train_metrics = train_one_epoch_clean(...)
    ...
```

Cách đơn giản hơn nhưng tốn chút memory: bật `requires_grad=True` cho
**toàn bộ** backbone ngay từ đầu, nhưng optimizer vẫn dùng đúng group.
Khi muốn "freeze" sớm, đặt `lr=0` cho group đó. Khi "unfreeze" chỉ cần
set lại `lr`:

```python
# Không rebuild optimizer, chỉ đổi LR theo epoch
def get_lr_multiplier(group_name: str, epoch: int, args) -> float:
    if group_name.startswith("backbone_stem") or group_name.startswith("backbone_c2"):
        return 1.0 if epoch >= args.unfreeze_shallow_epoch else 0.0
    if group_name.startswith("backbone_c3"):
        return 1.0 if epoch >= args.unfreeze_mid_epoch else 0.0
    if group_name.startswith("backbone_c4") or group_name.startswith("backbone_c5"):
        return 1.0 if epoch >= args.unfreeze_deep_epoch else 0.0
    return 1.0

# Trong loop:
for g in optimizer.param_groups:
    g["lr"] = g["initial_lr"] * get_lr_multiplier(g["name"], epoch, args)
```

Cách này gọn hơn vì không cần rebuild optimizer và scheduler; LR=0
cho group → param vẫn nhận gradient nhưng không update. Đây cũng
là cách tôi khuyên dùng.

## Bước 6 — Log để kiểm tra

Thêm vào training loop:

```python
if epoch == 1 or epoch % 5 == 0:
    print(model.encoder.describe())
    for g in optimizer.param_groups:
        n_trainable = sum(p.numel() for p in g["params"] if p.requires_grad)
        n_total = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:<30} lr={g['lr']:.2e}  trainable={n_trainable:,}/{n_total:,}")
```

Log này cực kỳ quan trọng để bạn biết lúc nào stage nào đang được
train. Nếu `trainable=0` cho nhóm `backbone_c2` ở epoch 5 → đúng theo
schedule. Nếu `trainable` khác 0 sớm hơn mong đợi → có bug trong
`set_epoch`.

## Gợi ý hyperparameter khởi đầu

Với PVTv2-B2 trên Kvasir (train 900 ảnh):

```bash
python train.py \
    --dataset glas \
    --file-path
    --encoder-name swinv2_large --use-pretrained \
    --image-size 256 256 --batch-size 4 --epochs 80 --warmup-epochs 5 \
    --encoder-lr 1e-4 --decoder-lr 3e-4 --adaptor-lr 3e-4 \
    --backbone-lr-decay 0.5 \
    --unfreeze-deep-epoch 0 --unfreeze-mid-epoch 8 --unfreeze-shallow-epoch 20 \
    --nl-stage-config-preset default \
    --enable-nested --nested-start-epoch 20 \
    --use-ema --patience 15 \
    --multi-scale-heads --use-cross-stage ----glas-val-ratio 0.15
```

Ý tưởng:
- Epoch 0-8: chỉ decoder + NL adaptor + backbone c4/c5 học.
  Backbone c2, c3 hoàn toàn frozen → giữ nguyên prior ImageNet.
- Epoch 8-20: thêm c3. Polyp semantic bắt đầu ảnh hưởng mid-level
  features.
- Epoch 20+: toàn bộ unfreeze với LLRD. Stem + c2 học rất chậm
  (lr=6.25e-6), c5 học bình thường (lr=1e-4). Decoder NL memory
  (nested_start_epoch) cũng bật lên.

## Kỳ vọng về metric

So với bản cũ, bạn nên thấy:

- **Early epoch (1-10)**: loss giảm chậm hơn (vì backbone bị freeze một phần)
  nhưng val IoU giữ được mức hợp lý vì prior không bị phá.
- **Mid training (10-30)**: val IoU tăng nhanh hơn khi c3 mở khóa.
- **Late training (30+)**: curve mượt hơn, ít spike xuống hơn vì stem/c2
  học rất chậm → tránh được "catastrophic forgetting" trên dataset nhỏ.
- **Generalization cross-dataset** (Kvasir → ETIS, ColonDB): đây mới là
  thứ đáng quan sát. LLRD + selective unfreeze được chứng minh nhiều
  lần trong paper transfer learning là giúp generalization tốt hơn.

Nếu bạn không thấy cải thiện trên cùng dataset nhưng thấy cải thiện
cross-dataset → đó là dấu hiệu kiến trúc đang làm đúng việc: giữ
general features và chỉ adapt task-specific layers.
