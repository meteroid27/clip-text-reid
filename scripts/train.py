import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clip_text_reid.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP Text-ReID (stage 1)")
    parser.add_argument("--dataset", default="MaulikMadhavi/CUHK-PEDES-processed")
    parser.add_argument("--save-dir", default=str(Path("checkpoints")))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-ctx", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    train(
        dataset_name=args.dataset,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_ctx=args.n_ctx,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
