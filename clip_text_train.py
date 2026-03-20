import os, math, random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.deterministic    = False
torch.backends.cudnn.benchmark        = True
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

try:
    import clip
    from datasets import load_dataset
except ImportError:
    os.system("pip install git+https://github.com/openai/CLIP.git datasets")
    import clip
    from datasets import load_dataset

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATASET_NAME  = "MaulikMadhavi/CUHK-PEDES-processed"
SAVE_DIR      = "/teamspace/studios/this_studio"

EPOCHS        = 20
LR            = 3.5e-4
BATCH_SIZE    = 512
TEMP_START    = 0.10
TEMP_END      = 0.07
N_CTX         = 8
WEIGHT_DECAY  = 1e-4
WARMUP_RATIO  = 0.1
GRAD_CLIP     = 1.0
USE_BF16      = True
TRAIN_RATIO   = 0.8
CAPTION_DROP  = 0.1
NUM_WORKERS   = 12
PREFETCH_FACTOR = 4
USE_COMPILE   = True

# ─────────────────────────────────────────────────────────────────────────────
# PERSON ID EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_pid(filename):
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    if '_' in stem:      return stem.split('_')[0]
    elif stem.isdigit(): return stem[:4]
    return stem

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
def expand_and_split(hf_ds, train_ratio=0.8, seed=42):
    cols      = hf_ds.column_names
    cap_col   = next((c for c in ['captions', 'caption', 'text'] if c in cols), None)
    fname_col = next((c for c in ['file_name', 'filename', 'image_name', 'name'] if c in cols), None)

    print("Expanding dataset...")
    expanded    = []
    pid_to_rows = defaultdict(list)

    for row_idx in tqdm(range(len(hf_ds)), desc="  Expanding"):
        item  = hf_ds[row_idx]
        fname = item[fname_col] if fname_col else f"row_{row_idx}"
        pid   = extract_pid(fname)
        caps  = item[cap_col]
        if isinstance(caps, str): caps = [caps]
        for cap in caps:
            exp_idx = len(expanded)
            expanded.append(dict(row_idx=row_idx, caption=cap,
                                 filename=fname, pid=pid))
            pid_to_rows[pid].append(exp_idx)

    all_pids  = sorted(pid_to_rows.keys())
    rng       = random.Random(seed)
    pids_shuf = all_pids.copy()
    rng.shuffle(pids_shuf)

    n_train    = int(len(pids_shuf) * train_ratio)
    train_pids = set(pids_shuf[:n_train])
    test_pids  = set(pids_shuf[n_train:])

    train_idx = [i for i, e in enumerate(expanded) if e['pid'] in train_pids]
    test_idx  = [i for i, e in enumerate(expanded) if e['pid'] in test_pids]

    tr_pids_sorted = sorted(train_pids)
    tr_pid2int     = {p: i for i, p in enumerate(tr_pids_sorted)}
    n_classes      = len(tr_pids_sorted)

    print(f"  Expanded       : {len(expanded)}")
    print(f"  Train persons  : {len(train_pids)} | rows: {len(train_idx)}")
    print(f"  Test  persons  : {len(test_pids)}  | rows: {len(test_idx)}")
    return expanded, train_idx, test_idx, tr_pid2int, n_classes


class ReIDDataset(Dataset):
    def __init__(self, hf_ds, expanded, indices, tr_pid2int,
                 transform=None, caption_drop=0.0):
        self.hf_ds        = hf_ds
        self.expanded     = expanded
        self.indices      = indices
        self.tr_pid2int   = tr_pid2int
        self.transform    = transform
        self.caption_drop = caption_drop
        cols = hf_ds.column_names
        self.img_col  = next((c for c in ['image', 'img'] if c in cols), None)
        self.pids     = [tr_pid2int[expanded[i]['pid']] for i in indices]
        self.image_cache = {}

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        e = self.expanded[self.indices[idx]]

        if e['row_idx'] in self.image_cache:
            img = self.image_cache[e['row_idx']].copy()
        else:
            item = self.hf_ds[e['row_idx']]
            img  = item[self.img_col]
            if not isinstance(img, Image.Image): img = Image.open(img)
            img  = img.convert('RGB')

        if self.transform: img = self.transform(img)

        caption = e['caption']
        if self.caption_drop > 0 and random.random() < self.caption_drop:
            caption = "a photo of a person"

        tokens = clip.tokenize(caption, truncate=True).squeeze(0)
        label  = self.tr_pid2int[e['pid']]
        return img, tokens, label

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
class IdentityPromptLearner(nn.Module):
    def __init__(self, clip_model, n_classes, n_ctx=4):
        super().__init__()
        dtype   = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        ctx = torch.empty(n_classes, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx, std=0.02)
        self.ctx             = nn.Parameter(ctx)
        self.token_embedding = clip_model.token_embedding
        self.n_ctx           = n_ctx
        self.dtype           = dtype
        self.n_classes       = n_classes
        template = clip.tokenize(["a photo of a person"]).squeeze(0)
        with torch.no_grad():
            emb = clip_model.token_embedding(template.unsqueeze(0)).squeeze(0)
        self.register_buffer("prefix",          emb[:1].unsqueeze(0))
        self.register_buffer("suffix",          emb[1+n_ctx:].unsqueeze(0))
        self.register_buffer("template_tokens", template.unsqueeze(0))

    def forward(self, labels):
        B   = labels.shape[0]
        ctx = self.ctx[labels]
        pre = self.prefix.expand(B, -1, -1)
        suf = self.suffix.expand(B, -1, -1)
        return torch.cat([pre, ctx, suf], dim=1).type(self.dtype)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer          = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final             = clip_model.ln_final
        self.text_projection      = clip_model.text_projection
        self.dtype                = clip_model.dtype

    def forward(self, prompts, tokenized_text, n_ctx):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        eot     = tokenized_text.argmax(dim=-1)
        shifted = torch.clamp(eot + n_ctx, max=x.shape[1] - 1)
        return x[torch.arange(x.shape[0]), shifted] @ self.text_projection


class CLIPReID(nn.Module):
    def __init__(self, n_classes, n_ctx=4):
        super().__init__()
        clip_model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
        clip_model.float()
        self.n_ctx     = n_ctx
        self.n_classes = n_classes
        self.image_encoder  = clip_model.visual
        self.prompt_learner = IdentityPromptLearner(clip_model, n_classes, n_ctx)
        self.text_encoder   = TextEncoder(clip_model)
        self._interp_pos(clip_model, 256, 128)
        self.bottleneck = nn.BatchNorm1d(512)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(512, n_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        for p in self.text_encoder.parameters():   p.requires_grad = False
        for p in self.image_encoder.parameters():  p.requires_grad = False
        for p in self.prompt_learner.token_embedding.parameters(): p.requires_grad = False
        for p in self.bottleneck.parameters():     p.requires_grad = False
        for p in self.classifier.parameters():     p.requires_grad = False

    def _interp_pos(self, clip_model, new_h, new_w):
        pos   = clip_model.visual.positional_embedding
        cls_p = pos[0:1]
        spa   = pos[1:].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
        spa   = F.interpolate(spa, size=(new_h // 16, new_w // 16),
                               mode='bicubic', align_corners=False)
        spa   = spa.permute(0, 2, 3, 1).reshape(-1, pos.shape[-1])
        clip_model.visual.positional_embedding = nn.Parameter(
            torch.cat([cls_p, spa], dim=0))

    @torch.no_grad()
    def encode_image_frozen(self, images):
        return F.normalize(self.image_encoder(images), dim=-1)

    def encode_text(self, labels):
        prompts = self.prompt_learner(labels)
        tokens  = self.prompt_learner.template_tokens.expand(
            labels.shape[0], -1).to(labels.device)
        feat = self.text_encoder(prompts, tokens, self.n_ctx)
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def encode_text_query(self, raw_tokens):
        B = raw_tokens.shape[0]
        x = self.text_encoder.transformer(
            (self.prompt_learner.token_embedding(raw_tokens).float() +
             self.text_encoder.positional_embedding.float()
             ).permute(1, 0, 2)
        ).permute(1, 0, 2)
        x    = self.text_encoder.ln_final(x).float()
        eot  = raw_tokens.argmax(dim=-1)
        feat = x[torch.arange(B), eot] @ self.text_encoder.text_projection
        return F.normalize(feat, dim=-1)

# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────
def contrastive_loss(img_feats, txt_feats, labels, temperature):
    sim_i2t    = img_feats @ txt_feats.t() / temperature
    sim_t2i    = txt_feats @ img_feats.t() / temperature
    same       = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    n_pos      = same.sum(dim=1, keepdim=True).clamp(min=1)
    soft_label = same / n_pos
    loss_i2t   = -(soft_label * F.log_softmax(sim_i2t, dim=1)).sum(dim=1).mean()
    loss_t2i   = -(soft_label * F.log_softmax(sim_t2i, dim=1)).sum(dim=1).mean()
    return (loss_i2t + loss_t2i) / 2

# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────
def build_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    ws = warmup_epochs * steps_per_epoch
    ts = total_epochs  * steps_per_epoch
    def lr_fn(step):
        if step < ws:
            return step / max(1, ws)
        prog = (step - ws) / max(1, ts - ws)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

def get_temperature(epoch, total_epochs, temp_start, temp_end):
    progress = min(epoch / total_epochs, 1.0)
    return temp_start - (temp_start - temp_end) * progress

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def train():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 65)
    print("  CLIP-ReID  —  Stage 1  (20 epochs, no early stopping)")
    print("=" * 65)
    print(f"  Device     : {DEVICE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  N_CTX      : {N_CTX}")
    print(f"  Encoders   : FROZEN")
    print("=" * 65 + "\n")

    print(f"Loading {DATASET_NAME} ...")
    raw_ds = load_dataset(DATASET_NAME)
    hf_ds  = raw_ds['train'] if 'train' in raw_ds else raw_ds[list(raw_ds.keys())[0]]

    expanded, train_idx, test_idx, tr_pid2int, n_classes = \
        expand_and_split(hf_ds, train_ratio=TRAIN_RATIO, seed=SEED)

    np.save(os.path.join(SAVE_DIR, "test_indices.npy"), np.array(test_idx))
    print(f"  Training identities : {n_classes}\n")

    transform = T.Compose([
        T.Resize((256, 128)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.3, hue=0.08),
        T.ToTensor(),
        T.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275)),
    ])

    train_ds = ReIDDataset(
        hf_ds, expanded, train_idx, tr_pid2int,
        transform=transform, caption_drop=CAPTION_DROP
    )

    print("  Pre-caching images into RAM...")
    cols    = hf_ds.column_names
    img_col = next((c for c in ['image', 'img'] if c in cols), None)
    for row_idx in tqdm(set(expanded[i]['row_idx'] for i in train_idx),
                        desc="  Caching"):
        item = hf_ds[row_idx]
        img  = item[img_col]
        if not isinstance(img, Image.Image): img = Image.open(img)
        train_ds.image_cache[row_idx] = img.convert('RGB')
    print(f"  Cached {len(train_ds.image_cache)} images\n")

    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True,
        drop_last=True,
        pin_memory=True,
        pin_memory_device="cuda",
    )
    print(f"  Train samples : {len(train_ds)} | batches/epoch: {len(loader)}\n")

    model = CLIPReID(n_classes=n_classes, n_ctx=N_CTX).to(DEVICE)

    if USE_COMPILE:
        print("  Compiling model (first batch slow — normal)...")
        model = torch.compile(model, mode="max-autotune")

    scaler    = GradScaler(enabled=USE_BF16)
    raw_model = model._orig_mod if USE_COMPILE else model

    trainable = [n for n, p in raw_model.named_parameters() if p.requires_grad]
    assert trainable == ['prompt_learner.ctx'], \
        f"Expected only ['prompt_learner.ctx'], got: {trainable}"
    print(f"  Trainable params : {raw_model.prompt_learner.ctx.numel():,}  (prompt_learner.ctx only)\n")

    optimizer = optim.AdamW(
        [raw_model.prompt_learner.ctx],
        lr=LR, weight_decay=WEIGHT_DECAY, fused=True
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=max(1, int(EPOCHS * WARMUP_RATIO)),
        total_epochs=EPOCHS,
        steps_per_epoch=len(loader)
    )

    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        raw_model.image_encoder.eval()
        raw_model.text_encoder.eval()
        raw_model.bottleneck.eval()
        raw_model.classifier.eval()

        temperature      = get_temperature(epoch, EPOCHS, TEMP_START, TEMP_END)
        total_train_loss = 0.0
        pbar = tqdm(loader, desc=f"S1 Ep {epoch:2d}/{EPOCHS}")

        for images, tokens, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16,
                          enabled=USE_BF16):
                img_feat = raw_model.encode_image_frozen(images)
                txt_feat = raw_model.encode_text(labels)
                loss     = contrastive_loss(img_feat, txt_feat,
                                            labels, temperature)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([raw_model.prompt_learner.ctx], GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                temp=f"{temperature:.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.1e}"
            )

        avg = total_train_loss / len(loader)
        print(f"\n  Ep {epoch:2d}/{EPOCHS}  loss={avg:.4f}  "
              f"temp={temperature:.3f}  lr={scheduler.get_last_lr()[0]:.2e}")

        save_ckpt = raw_model.state_dict()
        torch.save(save_ckpt, os.path.join(SAVE_DIR, "s1_last.pth"))

        if epoch % 5 == 0:
            torch.save(save_ckpt,
                       os.path.join(SAVE_DIR, f"s1_epoch{epoch}.pth"))
            print(f"  → Checkpoint: s1_epoch{epoch}.pth")

        if avg < best_loss:
            best_loss = avg
            torch.save(save_ckpt, os.path.join(SAVE_DIR, "s1_best.pth"))
            print(f"  → Best saved (loss={best_loss:.4f})")

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print(f"  Best loss  : {best_loss:.4f}")
    print(f"  s1_best.pth, s1_last.pth, test_indices.npy → {SAVE_DIR}")
    print("=" * 65)
    print("""
INFERENCE:
  model = CLIPReID(n_classes=1, n_ctx=8)
  ckpt  = torch.load('s1_best.pth', map_location='cpu')
  ckpt  = {k.replace('module.',''): v for k,v in ckpt.items()}
  model.load_state_dict(ckpt, strict=False)
  model.eval()

  img_feat = model.encode_image_frozen(images)
  tokens   = clip.tokenize([query], truncate=True)
  txt_feat = model.encode_text_query(tokens)
  sims     = (txt_feat @ img_feat.T)
""")


if __name__ == "__main__":
    train()