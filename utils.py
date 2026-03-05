"""
utils.py — Training utilities for LSST classification.

Public API
----------
seed_everything(seed)
EarlyStopping(patience, mode)         — callable: es(value) → bool
train_epoch(model, loader, ...)       → (loss, acc)
eval_epoch(model, loader, ...)        → (loss, acc, preds, targets)
compute_metrics(y_true, y_pred, le)   → (acc, f1, report, cm)
get_embeddings(model, loader, device) → (embeddings, labels)
save_logs(logs, path)
compute_class_weights(y, K)           → torch.FloatTensor(K)
FocalLoss                             — nn.Module
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

# alias
set_seed = seed_everything


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Class weights
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(y, num_classes=None, device=None):
    """
    Inverse-frequency class weights, normalised so mean = 1.
    Returns torch.FloatTensor of shape (K,) on CPU (or device if provided).
    """
    y  = np.asarray(y, dtype=int)
    K  = num_classes if num_classes is not None else int(y.max()) + 1
    counts = np.bincount(y, minlength=K).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / counts
    w = w / w.mean()
    t = torch.tensor(w, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class focal loss on logits.
    gamma > 0 reduces the relative loss for easy examples.
    Use WeightedRandomSampler for class balance; no alpha weighting here.
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma     = float(gamma)
        self.reduction = reduction

    def forward(self, logits, targets):
        logp  = F.log_softmax(logits, dim=1)
        pt    = torch.exp(logp).gather(1, targets.unsqueeze(1)).squeeze(1)
        logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss  = -((1.0 - pt) ** self.gamma) * logpt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Early stopping monitor.

    Usage
    -----
    es = EarlyStopping(patience=20, mode="max")
    if es(val_f1):    # True when patience is exhausted → stop training
        break

    es.step(val_f1)   # returns (stop: bool, improved: bool)
    """

    def __init__(self, patience: int = 20, mode: str = "max"):
        self.patience = int(patience)
        self.mode     = mode
        self.best     = None
        self.bad      = 0

    def step(self, value):
        """Returns (stop: bool, improved: bool)."""
        if self.best is None:
            self.best = value
            self.bad  = 0
            return False, True

        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value
            self.bad  = 0
            return False, True

        self.bad += 1
        stop = self.bad >= self.patience
        return stop, False

    def __call__(self, value) -> bool:
        """Callable shortcut — returns True if training should stop."""
        stop, _ = self.step(value)
        return stop


# ─────────────────────────────────────────────────────────────────────────────
# Training loop helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device,
                use_mixup: bool = False, mixup_alpha: float = 0.3):
    """
    Train one epoch. Loader yields (x, mask, y).

    Returns
    -------
    avg_loss : float
    accuracy : float
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    n          = 0

    for batch in loader:
        x, mask, y = batch
        x    = x.to(device)
        mask = mask.to(device)
        y    = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_mixup and mixup_alpha > 0:
            lam      = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx      = torch.randperm(x.size(0), device=device)
            x_mix    = lam * x    + (1.0 - lam) * x[idx]
            mask_mix = torch.minimum(mask, mask[idx])
            logits   = model(x_mix, mask_mix)
            loss     = lam * criterion(logits, y) + (1.0 - lam) * criterion(logits, y[idx])
        else:
            logits = model(x, mask)
            loss   = criterion(logits, y)

        # Skip batch if loss is NaN/Inf (guards against corrupted model states)
        if not math.isfinite(loss.item()):
            optimizer.zero_grad(set_to_none=True)
            n += len(y)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs          = len(y)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == y).sum().item()
        n          += bs

    return total_loss / max(1, n), correct / max(1, n)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """
    Evaluate model on one epoch. Loader yields (x, mask, y).

    Returns
    -------
    avg_loss : float
    accuracy : float
    preds    : np.ndarray (N,) int
    targets  : np.ndarray (N,) int
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    n          = 0
    all_preds  = []
    all_tgts   = []

    for batch in loader:
        x, mask, y = batch
        x    = x.to(device)
        mask = mask.to(device)
        y    = y.to(device)

        logits = model(x, mask)

        if criterion is not None:
            total_loss += criterion(logits, y).item() * len(y)

        preds = logits.argmax(1)
        correct    += (preds == y).sum().item()
        all_preds.append(preds.cpu().numpy())
        all_tgts.append(y.cpu().numpy())
        n += len(y)

    y_pred   = np.concatenate(all_preds) if all_preds else np.array([], dtype=int)
    y_true   = np.concatenate(all_tgts)  if all_tgts  else np.array([], dtype=int)
    avg_loss = total_loss / max(1, n)
    acc      = correct    / max(1, n)

    return avg_loss, acc, y_pred, y_true


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, le=None):
    """
    Accuracy, macro F1, classification report, confusion matrix.

    Parameters
    ----------
    y_true, y_pred : array-like int
    le             : LabelEncoder or None (for human-readable class names)

    Returns
    -------
    acc    : float
    f1     : float  (macro)
    report : str
    cm     : np.ndarray (K, K)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)

    K = max(int(y_true.max()), int(y_pred.max())) + 1
    target_names = ([str(c) for c in le.classes_]
                    if le is not None
                    else [str(i) for i in range(K)])

    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return acc, f1, report, cm


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction  (for t-SNE)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_embeddings(model, loader, device):
    """
    Extract feature embeddings. Loader yields (x, mask, y).

    Uses model.encode(x, mask) when available, else model(x, mask).

    Returns
    -------
    embeddings : np.ndarray (N, d)
    labels     : np.ndarray (N,)  int
    """
    model.eval()
    embs = []
    labs = []

    for batch in loader:
        x, mask, y = batch
        x    = x.to(device)
        mask = mask.to(device)

        if hasattr(model, "encode"):
            z = model.encode(x, mask)
        else:
            z = model(x, mask)

        arr = z.detach().cpu().numpy()
        arr = np.nan_to_num(arr, nan=0.0, posinf=100.0, neginf=-100.0)
        embs.append(arr)
        labs.append(y.numpy())

    return np.concatenate(embs), np.concatenate(labs)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def save_logs(logs: dict, path: str):
    """
    Save training logs dict {key: list_of_floats} to a .npz file.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    np.savez(path, **{k: np.array(v, dtype=np.float32) for k, v in logs.items()})
