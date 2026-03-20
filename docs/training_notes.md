# Training Notes

## Summary

This project trains a CLIP-based Text ReID model using prompt learning on top of a frozen CLIP ViT-B/16 backbone.

## Key behavior

- Uses `MaulikMadhavi/CUHK-PEDES-processed` by default.
- Expands multi-caption samples into caption-level rows.
- Splits train/test by person identity, not by caption row.
- Uses contrastive loss with multi-positive soft labels.
- Trains only `prompt_learner.ctx` and freezes other model parts.

## Saved outputs

- `s1_best.pth`
- `s1_last.pth`
- `s1_epoch{n}.pth` every 5 epochs
- `test_indices.npy`

## Important training defaults

- Epochs: 20
- Batch size: 512
- Prompt context tokens (`n_ctx`): 8
- Learning rate: 3.5e-4
- Temperature schedule: 0.10 -> 0.07
