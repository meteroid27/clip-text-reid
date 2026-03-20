# CLIP Text ReID

Training setup for text-to-image person re-identification using CLIP prompt learning (Stage 1).

## Repository structure

- `src/clip_text_reid/`: reusable training package
- `scripts/train.py`: command-line training entrypoint
- `checkpoints/`: saved model files (best/last/epoch)
- `docs/training_notes.md`: implementation and run notes
- `clip_text_train.py`: original standalone training script (kept for reference)

## Quick start

```bash
pip install -r requirements.txt
python scripts/train.py
```

## Training command (custom)

```bash
python scripts/train.py \
  --dataset MaulikMadhavi/CUHK-PEDES-processed \
  --save-dir checkpoints \
  --epochs 20 \
  --batch-size 512 \
  --n-ctx 8 \
  --lr 3.5e-4
```

## Notes

- Stage 1 keeps image/text encoders frozen and trains only prompt context vectors.
- Best checkpoint is saved as `s1_best.pth`.
- Dataset split is person-ID disjoint for train/test.

## Inference sketch

```python
import torch
import clip
from src.clip_text_reid.train import CLIPReID

model = CLIPReID(n_classes=1, n_ctx=8)
ckpt = torch.load("checkpoints/s1_best.pth", map_location="cpu")
ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
model.load_state_dict(ckpt, strict=False)
model.eval()
```
