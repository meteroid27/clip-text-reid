# Text-to-Image Person Re-ID — IRRA (CVPR 2023)

A faithful single-file implementation of **IRRA** — *Cross-Modal Implicit Relation Reasoning and Alignment for Text-to-Image Person Retrieval* (Jiang & Ye, CVPR 2023).

> **Reference:** [arxiv 2303.12501](https://arxiv.org/abs/2303.12501) · [Official repo](https://github.com/anosorae/IRRA)

---

## Repository Structure

```
train_irra.py       ← single-file IRRA implementation (train here)
requirements.txt    ← dependencies
README.md
LICENSE
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python train_irra.py
```

> **Default dataset:** `MaulikMadhavi/CUHK-PEDES-processed` (auto-downloaded from HuggingFace)
> **Default save dir:** `/teamspace/studios/this_studio` — change `SAVE_DIR` at the top of `train_irra.py` to your preferred path.

### 3. Resume training

Set these flags in `train_irra.py`:

```python
RESUME      = True
RESUME_PATH = "/path/to/last_model_irra.pth"
START_EPOCH = 7   # last completed epoch + 1
```

---

## Key Hyperparameters

| Parameter      | Value        | Note                               |
|----------------|--------------|------------------------------------|
| `BATCH_SIZE`   | 32           | × 2 gradient accum = 64 effective  |
| `EPOCHS`       | 30           | IRRA paper uses 60                 |
| `BASE_LR`      | 1e-5         | CLIP backbone learning rate        |
| `NEW_MODULE_LR`| 1e-4         | Cross-attn / MLM / classifier LR   |
| `WEIGHT_DECAY` | 4e-4         | IRRA exact setting                 |
| `N_CTX`        | 8            | Learnable prompt context tokens    |
| `SDM_SIGMA`    | 0.01         | Temperature for SDM loss           |
| `MLM_PROB`     | 0.15         | BERT-style masking probability     |

---

## Inference

```python
import torch
import clip
from train_irra import IRRA, N_CTX

# Load model
model = IRRA(n_ctx=N_CTX, n_classes=1)   # n_classes=1 for inference
ckpt  = torch.load("best_model_irra.pth", map_location="cpu")
ckpt  = {k.replace("module.", ""): v for k, v in ckpt.items()}
model.load_state_dict(ckpt, strict=False)
model.eval()

# Encode image
img_feat, _ = model._encode_image(images)   # [B, 512], L2-normalised

# Encode text query
tokens   = clip.tokenize(["a person wearing a red jacket"], truncate=True)
txt_feat, _ = model._encode_text(tokens)    # [B, 512], L2-normalised

# Compute similarity
sims = txt_feat @ img_feat.T               # [1, B]
```

---

## Architecture Overview

```
Input Image ──► CLIP ViT-B/16 ──► global_feat  ─────────────────────► SDM Loss ◄─┐
                                 └─► patch_feats ──┐                  ID Loss  ◄─┤
                                                    ▼                              │
Input Text ──► CLIP Transformer ──► global_feat ───────────────────────────────────┘
               (with n_ctx=8     └─► token_feats ──► Cross-Attn ──► MLM Head ──► MLM Loss
                prompt tokens)         (masked)      (text→patches)
```

---

## Loss Functions

| Loss | Formula | Weight |
|------|---------|--------|
| `L_SDM` | KL( label_dist ‖ softmax(sim/σ) ) bidirectional | 1.0 |
| `L_ID`  | CrossEntropy(img_logits, labels) + CrossEntropy(txt_logits, labels) | 1.0 |
| `L_MLM` | CrossEntropy over masked token positions | 1.0 |
| **Total** | `L_SDM + L_ID + L_MLM` | — |

---

## Checkpoints Saved

| File | Description |
|------|-------------|
| `best_model_irra.pth` | Best model (lowest total loss) |
| `last_model_irra.pth` | Most recent epoch checkpoint |
| `test_indices_irra.npy` | Person-disjoint test split indices |
