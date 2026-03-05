"""
data/lsst_dataset.py — LSST dataset loading, preprocessing and DataLoaders.

Public API
----------
load_lsst()                  → X_train_raw, y_train, X_test_raw, y_test, le
preprocess(X_raw)            → X_norm, mask
compute_class_weights(y, K)  → torch.FloatTensor(K)
get_weighted_sampler(y, K)   → WeightedRandomSampler
get_dataloaders(...)         → train_loader, val_loader, test_loader, X_test_norm, test_mask

Each loader yields (x, mask, y):
    x    : (T, C) float32 — normalised time series
    mask : (T, C) float32 — 1 = observed, 0 = was NaN
    y    : ()     long    — class index in [0..13]
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_lsst():
    """
    Load LSST via tslearn, label-encode to 0..13.

    Returns
    -------
    X_train_raw : (N, T, C) float32  — raw, may contain NaNs
    y_train     : (N,)      int
    X_test_raw  : (M, T, C) float32
    y_test      : (M,)      int
    le          : LabelEncoder (le.classes_ = original str labels)
    """
    from tslearn.datasets import UCR_UEA_datasets

    ds = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = ds.load_dataset("LSST")

    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train.ravel()).astype(int)
    y_test_enc  = le.transform(y_test.ravel()).astype(int)

    print(f"  LSST loaded — train: {X_train.shape}, test: {X_test.shape}")
    print(f"  Shape: (N, T={X_train.shape[1]}, C={X_train.shape[2]})")
    print(f"  Classes: {le.classes_.tolist()}")
    nan_pct = np.isnan(X_train).mean() * 100
    print(f"  NaN rate (train): {nan_pct:.1f}%")

    return X_train, y_train_enc, X_test, y_test_enc, le


# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _fill_nans(X):
    """
    In-place NaN fill: forward fill → backward fill → zero fill.
    X : (N, T, C) float32
    """
    N, T, C = X.shape
    for n in range(N):
        for c in range(C):
            s = X[n, :, c]
            if not np.any(np.isnan(s)):
                continue
            # forward fill
            last = np.nan
            for t in range(T):
                if not np.isnan(s[t]):
                    last = s[t]
                elif not np.isnan(last):
                    s[t] = last
            # backward fill
            last = np.nan
            for t in range(T - 1, -1, -1):
                if not np.isnan(s[t]):
                    last = s[t]
                elif not np.isnan(last):
                    s[t] = last
            # zero fill remaining (all-NaN channel)
            s[np.isnan(s)] = 0.0
            X[n, :, c] = s
    return X


def preprocess(X_raw):
    """
    NaN fill + per-sample per-channel z-score normalization.

    Parameters
    ----------
    X_raw : (N, T, C) float32

    Returns
    -------
    X_norm : (N, T, C) float32
    mask   : (N, T, C) float32  — 1=originally observed, 0=was NaN
    """
    X    = X_raw.copy().astype(np.float32)
    mask = (~np.isnan(X)).astype(np.float32)   # (N, T, C)

    _fill_nans(X)

    # Per-sample per-channel z-score  (axis=1 = time axis)
    mu  = X.mean(axis=1, keepdims=True)          # (N, 1, C)
    sig = X.std(axis=1, keepdims=True) + 1e-8    # (N, 1, C)
    X_norm = (X - mu) / sig

    return X_norm.astype(np.float32), mask


# ─────────────────────────────────────────────────────────────────────────────
# 3. Class imbalance helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(y, num_classes=None):
    """
    Inverse-frequency class weights, normalised so mean weight = 1.

    Parameters
    ----------
    y           : array-like of int labels in [0..K-1]
    num_classes : int or None (auto-detected)

    Returns
    -------
    torch.FloatTensor of shape (K,)  — on CPU
    """
    y  = np.asarray(y, dtype=int)
    K  = num_classes if num_classes is not None else int(y.max()) + 1
    counts = np.bincount(y, minlength=K).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def get_weighted_sampler(y, num_classes=None):
    """
    WeightedRandomSampler for class-balanced batches.

    Parameters
    ----------
    y           : array-like of int labels
    num_classes : int or None

    Returns
    -------
    WeightedRandomSampler
    """
    y = np.asarray(y, dtype=int)
    K = num_classes if num_classes is not None else int(y.max()) + 1
    counts   = np.bincount(y, minlength=K).astype(np.float64)
    counts   = np.maximum(counts, 1.0)
    sample_w = torch.tensor((1.0 / counts)[y], dtype=torch.double)
    return WeightedRandomSampler(sample_w, len(sample_w), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dataset
# ─────────────────────────────────────────────────────────────────────────────

class LSSTPatchDataset(Dataset):
    """
    Yields (x, mask, y) tuples.

    Parameters
    ----------
    X           : (N, T, C) float32 — already normalised
    y           : (N,)      int
    mask        : (N, T, C) float32 or None (defaults to all-ones)
    augment     : bool — apply data augmentation on __getitem__
    jitter_sigma, scale_range, channel_drop_p, time_warp_p : augmentation params
    """

    def __init__(self, X, y, mask=None, augment=False,
                 jitter_sigma=0.03, scale_range=(0.9, 1.1),
                 channel_drop_p=0.1, time_warp_p=0.2):
        self.X    = torch.tensor(X, dtype=torch.float32)
        self.y    = torch.tensor(y, dtype=torch.long)
        self.mask = (torch.tensor(mask, dtype=torch.float32)
                     if mask is not None
                     else torch.ones_like(self.X))

        self.augment        = augment
        self.jitter_sigma   = jitter_sigma
        self.scale_range    = scale_range
        self.channel_drop_p = channel_drop_p
        self.time_warp_p    = time_warp_p

    def __len__(self):
        return len(self.y)

    def _augment(self, x, mask):
        """
        x    : (T, C) tensor
        mask : (T, C) tensor
        Returns augmented (x, mask).
        """
        # 1) Jitter: additive Gaussian noise
        if self.jitter_sigma > 0:
            x = x + torch.randn_like(x) * self.jitter_sigma

        # 2) Per-channel random scaling
        lo, hi = self.scale_range
        if lo != hi:
            scale = torch.empty(x.shape[1]).uniform_(lo, hi)   # (C,)
            x = x * scale.unsqueeze(0)                          # (T, C)

        # 3) Channel dropout: zero-out entire channels
        if self.channel_drop_p > 0:
            drop = torch.rand(x.shape[1]) < self.channel_drop_p  # (C,)
            if drop.any():
                x    = x.clone()
                mask = mask.clone()
                x[:, drop]    = 0.0
                mask[:, drop] = 0.0

        # 4) Time warp (simplified: reverse a random short segment)
        if self.time_warp_p > 0 and torch.rand(1).item() < self.time_warp_p:
            T   = x.shape[0]
            seg = max(2, T // 6)
            start = torch.randint(0, T - seg + 1, (1,)).item()
            x    = x.clone()
            mask = mask.clone()
            x[start:start + seg]    = x[start:start + seg].flip(0)
            mask[start:start + seg] = mask[start:start + seg].flip(0)

        return x, mask

    def __getitem__(self, idx):
        x    = self.X[idx].clone()
        mask = self.mask[idx].clone()
        y    = self.y[idx]

        if self.augment:
            x, mask = self._augment(x, mask)

        return x, mask, y


# ─────────────────────────────────────────────────────────────────────────────
# 5. DataLoaders factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    X_train_raw, y_train, X_test_raw, y_test,
    val_ratio=0.2,
    batch_size=64,
    aug_kwargs=None,
    random_state=42,
    use_weighted_sampler=True,
    num_classes=None,
):
    """
    Build train / val / test DataLoaders from raw LSST arrays.

    Parameters
    ----------
    X_train_raw, y_train : training set (raw, may contain NaNs)
    X_test_raw,  y_test  : test set
    val_ratio            : fraction of training data used for validation
    batch_size           : batch size for all loaders
    aug_kwargs           : dict passed to LSSTPatchDataset for augmentation
                           (jitter_sigma, scale_range, channel_drop_p, time_warp_p)
    random_state         : reproducibility seed for the train/val split
    use_weighted_sampler : if True, use WeightedRandomSampler to balance classes
    num_classes          : explicit number of classes (auto-detected if None)

    Returns
    -------
    train_loader, val_loader, test_loader,
    X_test_norm (np.ndarray),
    test_mask   (np.ndarray)
    """
    if aug_kwargs is None:
        aug_kwargs = {}

    # ── Preprocessing ────────────────────────────────────────────────────────
    X_train_norm, train_mask = preprocess(X_train_raw)
    X_test_norm,  test_mask  = preprocess(X_test_raw)

    # ── Stratified train/val split ───────────────────────────────────────────
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=random_state
    )
    tr_idx, va_idx = next(sss.split(X_train_norm, y_train))

    X_tr, y_tr, m_tr = X_train_norm[tr_idx], y_train[tr_idx], train_mask[tr_idx]
    X_va, y_va, m_va = X_train_norm[va_idx], y_train[va_idx], train_mask[va_idx]

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = LSSTPatchDataset(X_tr, y_tr, m_tr, augment=True,  **aug_kwargs)
    val_ds   = LSSTPatchDataset(X_va, y_va, m_va, augment=False)
    test_ds  = LSSTPatchDataset(X_test_norm, y_test, test_mask, augment=False)

    # ── Sampler ──────────────────────────────────────────────────────────────
    sampler = None
    if use_weighted_sampler:
        K = num_classes if num_classes is not None else (int(y_tr.max()) + 1)
        sampler = get_weighted_sampler(y_tr, num_classes=K)

    # ── Loaders ──────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        drop_last=False,
    )
    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Split — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")
    if use_weighted_sampler:
        print(f"  WeightedRandomSampler ON (K={K})")

    return train_loader, val_loader, test_loader, X_test_norm, test_mask
