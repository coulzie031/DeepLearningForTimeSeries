"""
train_baseline.py — InceptionTime on LSST (baseline v2)

Objectif: meilleure Macro-F1 sous fort déséquilibre.

Stratégie v2 (recommandée):
  - WeightedRandomSampler = batches ~ équilibrés
  - PAS de class weights (évite double-compensation)
  - PAS de label smoothing
  - Focal Loss (gamma=2) pour réduire l'impact des exemples faciles
  - Warmup + Cosine LR (plus stable)
  - Early stopping sur Macro F1

Usage:  python train_baseline.py
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))

from data.lsst_dataset import load_lsst, get_dataloaders
from models.inception_time import InceptionTime
from utils import EarlyStopping, eval_epoch, compute_metrics, save_logs

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

CFG = {
    # Data
    "val_ratio":            0.2,
    "batch_size":           64,
    "random_state":         42,
    "use_weighted_sampler": True,   # v2 = True

    # Augmentation (plus légère, sinon ça peut noiser trop un dataset court T=36)
    "jitter_sigma":         0.03,
    "scale_range":          (0.9, 1.1),
    "channel_drop_p":       0.1,
    "time_warp_p":          0.2,

    # Model
    "nb_filters":           32,
    "kernel_sizes":         (9, 19, 39),
    "n_blocks":             2,
    "dropout":              0.2,

    # Training
    "lr":                   1e-3,
    "weight_decay":         1e-4,
    "n_epochs":             200,
    "grad_clip":            1.0,

    # Focal Loss
    "focal_gamma":          2.0,

    # Warmup + Cosine
    "warmup_epochs":        5,
    "min_lr":               1e-5,

    # Early stopping
    "patience":             25,

    # Paths
    "results_dir":          "results",
    "checkpoint":           "results/best_inception_time_v2.pt",
    "logs_path":            "results/logs_baseline_v2.npz",
}

NUM_CLASSES = 14
N_CHANNELS  = 6


# ─────────────────────────────────────────────────────────────
# Loss: Focal (sans alpha/weights pour éviter double-compensation)
# ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss multi-classe (softmax).
    - gamma > 0 : réduit le poids des exemples faciles.
    - Pas d'alpha ici (on a déjà un sampler équilibré).
    """
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (B, K), targets: (B,)
        logp = F.log_softmax(logits, dim=1)                  # (B, K)
        p = torch.exp(logp)                                  # (B, K)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)    # (B,)
        logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        loss = -((1.0 - pt) ** self.gamma) * logpt           # (B,)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────
# Scheduler: Warmup + Cosine
# ─────────────────────────────────────────────────────────────

def lr_warmup_cosine(epoch, base_lr, min_lr, warmup_epochs, max_epochs):
    # epoch starts at 1
    if epoch <= warmup_epochs:
        return base_lr * (epoch / max(1, warmup_epochs))
    # cosine from base_lr to min_lr
    t = (epoch - warmup_epochs) / max(1, (max_epochs - warmup_epochs))
    cos = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_lr + (base_lr - min_lr) * cos


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(CFG["results_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  BASELINE v2 — InceptionTime on LSST (sampler + focal)")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── 1) Data
    X_train_raw, y_train, X_test_raw, y_test, le = load_lsst()

    aug_kwargs = {
        "jitter_sigma":   CFG["jitter_sigma"],
        "scale_range":    CFG["scale_range"],
        "channel_drop_p": CFG["channel_drop_p"],
        "time_warp_p":    CFG["time_warp_p"],
    }

    (train_loader, val_loader, test_loader,
     X_test_norm, X_test_mask) = get_dataloaders(
        X_train_raw, y_train, X_test_raw, y_test,
        val_ratio=CFG["val_ratio"],
        batch_size=CFG["batch_size"],
        aug_kwargs=aug_kwargs,
        random_state=CFG["random_state"],
        use_weighted_sampler=CFG["use_weighted_sampler"],
    )

    # ── 2) Model
    model = InceptionTime(
        n_channels=N_CHANNELS,
        num_classes=NUM_CLASSES,
        nb_filters=CFG["nb_filters"],
        kernel_sizes=CFG["kernel_sizes"],
        n_blocks=CFG["n_blocks"],
        dropout=CFG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: InceptionTime  |  Parameters: {n_params:,}\n")

    # ── 3) Loss / opt
    criterion = FocalLoss(gamma=CFG["focal_gamma"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    early_stop = EarlyStopping(patience=CFG["patience"], mode="max")

    # ── 4) Loop
    logs = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_val_f1 = 0.0
    best_epoch  = 0

    header = (f"{'Ep':>5}  {'TrLoss':>7}  {'TrAcc':>6}  "
              f"{'VLoss':>7}  {'VAcc':>6}  {'VF1':>6}  {'LR':>8}")
    print(header)
    print("-" * len(header))

    for epoch in range(1, CFG["n_epochs"] + 1):
        # set LR (warmup+cosine)
        lr_now = lr_warmup_cosine(
            epoch=epoch,
            base_lr=CFG["lr"],
            min_lr=CFG["min_lr"],
            warmup_epochs=CFG["warmup_epochs"],
            max_epochs=CFG["n_epochs"],
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # --- train ---
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optimizer.step()

            tr_loss += loss.item() * len(y)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += len(y)

        tr_loss /= max(1, tr_total)
        tr_acc = tr_correct / max(1, tr_total)

        # --- val ---
        vl_loss, vl_acc, vl_preds, vl_targets = eval_epoch(model, val_loader, criterion, device)
        vl_f1 = f1_score(vl_targets, vl_preds, average="macro", zero_division=0)

        logs["train_loss"].append(tr_loss)
        logs["val_loss"].append(vl_loss)
        logs["train_acc"].append(tr_acc)
        logs["val_acc"].append(vl_acc)
        logs["val_f1"].append(vl_f1)

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_epoch  = epoch
            torch.save(model.state_dict(), CFG["checkpoint"])

        print(f"{epoch:>5d}  {tr_loss:>7.4f}  {tr_acc:>6.4f}  "
              f"{vl_loss:>7.4f}  {vl_acc:>6.4f}  {vl_f1:>6.4f}  {lr_now:>8.2e}")

        if early_stop(vl_f1):
            print(f"\n  Early stopping at epoch {epoch}.")
            break

    print(f"\n  Best val F1: {best_val_f1:.4f}  (epoch {best_epoch})")

    # ── 5) Test
    print("\nLoading best checkpoint...")
    model.load_state_dict(torch.load(CFG["checkpoint"], map_location=device, weights_only=True))

    te_loss, te_acc, te_preds, te_targets = eval_epoch(model, test_loader, criterion, device)
    acc, f1, report, cm = compute_metrics(te_targets, te_preds, le)

    print(f"\n{'='*60}")
    print(f"  BASELINE v2 TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"\n{report}")

    # ── 6) Save
    save_logs(logs, CFG["logs_path"])
    rd = CFG["results_dir"]
    np.save(f"{rd}/baseline_preds.npy",   te_preds)
    np.save(f"{rd}/baseline_targets.npy", te_targets)
    np.save(f"{rd}/baseline_cm.npy",      cm)

    with open(f"{rd}/results_baseline.txt", "w") as f:
        f.write(f"Baseline: InceptionTime (sampler + focal)\n")
        f.write(f"Best val F1   : {best_val_f1:.4f} (epoch {best_epoch})\n")
        f.write(f"Test accuracy : {acc:.4f}\n")
        f.write(f"Test macro F1 : {f1:.4f}\n\n")
        f.write(report)

    print(f"\n  Logs    → {CFG['logs_path']}")
    print(f"  Model   → {CFG['checkpoint']}")
    print(f"  Run evaluate.py to generate figures.\n")
    return acc, f1, logs


if __name__ == "__main__":
    main()