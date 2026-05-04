# Kiến trúc Model - PolyMemnet

## 📁 File Structure (sau khi dọn dẹp)

```
model/
├── advanced_polymemnet.py          # AdvancedPolyMemnet (tích hợp 6 enhancements)
├── advanced_modules.py             # CrossStageModulator, HierarchicalPrototypeBank, MetaOptimizer
├── multi_scale_heads.py            # Multi-scale heads
├── backbones/
│   ├── strong_baseline.py          # ✅ Baseline encoder + FPNDecoder + StrongBaselinePolypModel
│   ├── cms_decoder.py              # ✅ CMSDecoder với hierarchical prototypes
│   └── self_modifying_encoder_cms.py  # ✅ CMSSelfModifyingEncoder (CMS-aware)
└── [DELETED]
    ├── self_modifying.py           # ❌ Old version (unused)
    ├── self_modifying_encoder.py  # ❌ Old version (unused)
    └── self_modifying_encoder_minimal.py  # ❌ Minimal version (unused)
```

---

## 🎯 Two Main Architectures

### 1. StrongBaselinePolypModel (Baseline)
```
Input → Encoder(backbone) → c2-c5
       ↓
    FPNDecoder (BiFPN + SE)
       ↓
    coarse_logits + aux_feat
       ↓
    SafeNestedResidualRefiner
       ↓
    refined_logits
       ↓
    Upsample → seg_head
```

**Components:**
- **Encoder**: 4 variants (TinyConvNeXt, Torchvision ConvNeXt, PVTv2, SwinV2)
- **Decoder**: FPNDecoder với BiFPN weighted fusion + SE channel attention
- **Refiner**: SafeNestedResidualRefiner (dual memory banks: fast + slow)
- **Loss**: StrongBaselineLoss (7 components)

---

### 2. AdvancedPolyMemnet (Full Featured)

```
Input → CMSSelfModifyingEncoder (cross-stage + adaptive)
       ↓
    c2-c5 (adapted)
       ↓
    CMSDecoder (hierarchical prototypes + cross-attention)
       ↓
    fused + decoder_info
       ↓
    [no separate refiner - integrated]
       ↓
    seg_head
       ↓
    EnhancedStrongBaselineLoss (+6 enhancements)
```

**6 Enhancements:**
1. **Cross-Stage Modulation**: Attention between encoder stages
2. **Adaptive CMS**: Dynamic inner loop steps/lr based on surprise
3. **Meta Inner Optimizer**: Learnable optimizer replaces SGD
4. **Hierarchical Prototypes**: Tree-structured memory bank (3 levels)
5. **MC Dropout**: Uncertainty estimation at inference
6. **Enhanced Loss**: Contrastive, consistency, sparsity, quality regularization

---

## 🔧 Encoder Deep Dive

### Backbone Options

| Backbone | Source | Variants | Output Channels | Notes |
|----------|--------|----------|-----------------|-------|
| TinyConvNeXt | Custom | 1 | [64,128,320,512] | Lightweight, from scratch |
| Torchvision ConvNeXt | torchvision | tiny/small/base/large | varies | Pretrained available |
| PVTv2 | timm | b2, b5 | [64,128,320,512] | With offline fallback |
| SwinV2 | timm | tiny/small/base/large | varies | Requires img_size |

All return 4 stages: c2 (1/4), c3 (1/8), c4 (1/16), c5 (1/32)

---

### CMSSelfModifyingEncoder (CMS-aware)

**Per-Stage Configuration (CMS Principle):**

| Stage | Mode | inner_steps | inner_lr | surprise_type | persist_momentum | position_aware |
|-------|------|-------------|----------|---------------|------------------|----------------|
| c2 | none | 0 | - | - | - | - |
| c3 | light | 0 | - | - | - | - |
| c4 | full | 3 | 5e-3 | full | True | True |
| c5 | full | 6 | 2e-2 | full | True | True |

**Features:**
- **Progressive Unfreeze**:
  - epoch 0: unfreeze c4,c5 (deep)
  - epoch 8: unfreeze c3 (mid)
  - epoch 16: unfreeze c2 + stem (shallow)
- **LLRD** (Layer-wise LR Decay): backbone_lr_decay=0.5
  - stem: lr * decay^4
  - c2: lr * decay^3
  - c3: lr * decay^2
  - c4: lr * decay^1
  - c5: lr * decay^0
- **Cross-Stage Modulation** (optional): Attention between encoder stages
- **Adaptive CMS** (optional): Dynamic inner parameters

---

## 🎨 Decoder Deep Dive

### FPNDecoder (Baseline)

**Architecture:**
```
c2, c3, c4, c5
  ↓
Lateral projections (1x1 conv) + SE attention
  ↓
BiFPN Top-Down (p5→p4→p3→p2)
  ↓
BiFPN Bottom-Up (p2→p3→p4→p5)
  ↓
Smoothing (ConvBNAct per level)
  ↓
Concat [s2,s3,s4,s5]
  ↓
2x ConvBNAct
  ↓
fused + s5 (aux)
```

**BiFPN**: Learnable weighted fusion with `w = relu(w) / (sum(w) + eps)`

---

### CMSDecoder (Advanced)

**Features:**
- HierarchicalPrototypeBank: 3 levels (fast, medium, slow)
- Cross-Attention per level: each level attends to different encoder stages
- Memory gate: controls context mixing from prototype levels
- Returns `decoder_info` with `proto_cache` for refiner

**Prototype Bank:**
- Level 1 (fast): momentum 0.03, 8 prototypes
- Level 2 (medium): momentum 0.03, 8 prototypes
- Level 3 (slow): momentum 0.0075, 8 prototypes

---

### SafeNestedResidualRefiner

**Dual Memory Banks:**
```
fast_prototypes: [K, D]  (momentum=0.03)
slow_prototypes: [K, D]  (momentum=0.0075)
```

**Pipeline:**
1. Compute token from query feat (weighted by probs + uncertainty)
2. Spatial attention with both banks separately
3. Mix contexts: `context = mix*fast + (1-mix)*slow`
4. Uncertainty gate: `gate = sigmoid(mlp([feat, uncertainty]))`
5. Residual: `refined = coarse + scale * gate * residual_head([feat, context, uncertainty])`

**Update** (called every batch):
- Bootstrap inactive slots
- Momentum update: `bank[idx] = (1-momentum)*bank[idx] + momentum*normalized(token)`

---

## ⚖️ Loss Functions

### StrongBaselineLoss (7 components)

Weighted sum:
1. `bce_loss`: pixel-wise BCE
2. `dice_loss`: soft + hard dice
3. `lovasz_loss`: mIoU surrogate
4. `aux_loss`: from aux head (@1/8)
5. `boundary_loss`: focus on edges
6. `consistency_loss`: nested stage consistency
7. `sparsity_loss`: gate sparsity regularization

---

### EnhancedStrongBaselineLoss (+6)

Additional components:
1. `prototype_contrastive_weight`: encourage diverse prototypes
2. `inner_consistency_weight`: nested info consistency
3. `gate_sparsity_weight`: sparse gating
4. `memory_quality_weight`: quality threshold regularization

---

## 🔄 Training Flow

```python
# Parameter groups example
{
    "backbone_stem": lr * decay^4,
    "backbone_c2": lr * decay^3,
    "backbone_c3": lr * decay^2,
    "backbone_c4": lr * decay^1,
    "backbone_c5": lr * decay^0,
    "adaptor_outer_stage_X": adaptor_lr,
    "modifier_inner_stage_X": adaptor_lr * 0.5,
    "pre_norms": adaptor_lr,
}
```

---

## 🚀 Usage Examples

### StrongBaseline (Simple)
```python
model = StrongBaselinePolypModel(
    encoder_name="convnext_tiny",
    enable_nested=True,
    nested_prototypes=8,
)
```

### AdvancedPolyMemnet (Full)
```python
model = AdvancedPolyMemnet(
    encoder_name="pvtv2_b2",
    enable_nested=True,
    use_cross_stage_encoder=True,
    use_adaptive_cms=True,
    use_advanced_decoder=True,  # CMSDecoder
    use_hierarchical_prototypes=True,
    use_mc_dropout=True,
    use_enhanced_loss=True,
)
```

---

## 📊 Key Differences

| Feature | StrongBaseline | AdvancedPolyMemnet |
|---------|---------------|-------------------|
| Encoder | Static backbone | CMS-aware self-modifying |
| Decoder | FPNDecoder (BiFPN) | CMSDecoder (hierarchical) |
| Refiner | SafeNestedResidualRefiner | Integrated in decoder |
| Memory | Dual fast/slow banks | Hierarchical 3-level |
| Loss | 7 components | 13 components |
| Inner Loop | None | Per-sample adaptation |
| Params | ~50M | ~70-80M |
| Speed | Fast | Slower (2-3x) |

---

## 📝 Notes

- All self-modifying encoders use `torch.func.functional_call` for clean autograd
- Inner loop runs with `torch.enable_grad()` even at eval for test-time adaptation
- Prototype updates happen outside backward: `model.update_nested_prototypes(nested_cache)`
- CMSSelfModifyingBlock saves/restores modifier weights per forward (stateless inner loop)
- Cross-stage modulation applied **before** per-stage modification
- Progressive unfreeze controlled via `encoder.set_epoch(epoch)`

---

Generated from code analysis. See `model_architecture.html` for visual diagrams.
