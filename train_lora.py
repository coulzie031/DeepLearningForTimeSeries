"""
train_lora.py — MOMENT fine-tuning with LoRA (PEFT) instead of direct unfreeze.

LoRA adds low-rank adapter matrices to T5 attention (q, v layers):
  - ~1M trainable params vs 26M for full unfreeze
  - Stable training — no NaN (pre-trained weights never modified directly)
  - PEFT: Parameter-Efficient Fine-Tuning (Hu et al., ICLR 2022)

Pipeline identical to train_competitor.py except Step B uses LoRA.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))

from data.lsst_dataset import (
    load_lsst, get_dataloaders, preprocess,
    compute_class_weights, get_weighted_sampler
)
from models.moment_classifier import MOMENTClassifier, PatchTSTClassifier
from models.inception_time import InceptionTime
from utils import (
    seed_everything, EarlyStopping, train_epoch, eval_epoch, compute_metrics,
    get_embeddings, save_logs,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    # Data
    "val_ratio":            0.2,
    "batch_size":           32,
    "random_state":         42,
    "imbalance_strategy":   "sampler_only",

    # Augmentation
    "jitter_sigma":         0.03,
    "scale_range":          (0.9, 1.1),
    "channel_drop_p":       0.15,
    "time_warp_p":          0.2,

    # Loss
    "label_smoothing":      0.1,
    "use_mixup":            True,
    "mixup_alpha":          0.3,

    # LoRA config (replaces phase 1 + phase 2)
    "lora_rank":            8,
    "lora_alpha":           32,
    "lora_target_modules":  ["q", "v"],  # T5 attention Q and V matrices
    "lora_dropout":         0.05,
    "lora_lr":              1e-4,        # LoRA adapter LR
    "lora_head_lr":         1e-3,        # classification head LR
    "lora_epochs":          80,
    "lora_patience":        20,

    # Additional ensemble members (unchanged from train_competitor.py)
    "use_inception_large":  True,
    "inception_epochs":     200,
    "ptst_epochs":          150,
    "ptst_lr":              5e-4,
    "ptst_patience":        20,
    "rocket_n_kernels":     10000,
    "mlp_hidden":           256,
    "mlp_dropout":          0.3,
    "mlp_epochs":           100,
    "mlp_lr":               1e-3,
    "mlp_patience":         15,

    "results_dir":          "results",
}

NUM_CLASSES = 14
N_CHANNELS  = 6
SEQ_LEN     = 36


# ─────────────────────────────────────────────────────────────────────────────
# Feature MLP (same as train_competitor.py)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, num_classes=14, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x, mask=None):
        return self.net(x)

    def encode(self, x, mask=None):
        for layer in list(self.net.children())[:-1]:
            x = layer(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (same as train_competitor.py)
# ─────────────────────────────────────────────────────────────────────────────

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def train_model(model, train_loader, val_loader, optimizer, criterion,
                scheduler, early_stop, n_epochs, device, logs,
                use_mixup=False, mixup_alpha=0.3):
    best_val_f1 = 0.0
    best_state  = None

    header = (f"{'Epoch':>6}  {'TrLoss':>8}  {'TrAcc':>7}  "
              f"{'VLoss':>8}  {'VAcc':>7}  {'VF1':>7}")
    print(header)
    print("-" * len(header))

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_mixup=use_mixup, mixup_alpha=mixup_alpha,
        )
        vl_loss, vl_acc, vl_preds, vl_targets = eval_epoch(
            model, val_loader, criterion, device
        )
        vl_f1 = f1_score(vl_targets, vl_preds, average="macro", zero_division=0)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(vl_f1)
            else:
                scheduler.step()

        logs["train_loss"].append(tr_loss)
        logs["val_loss"].append(vl_loss)
        logs["train_acc"].append(tr_acc)
        logs["val_acc"].append(vl_acc)
        if "val_f1" in logs:
            logs["val_f1"].append(vl_f1)

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"{epoch:>6d}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  "
              f"{vl_loss:>8.4f}  {vl_acc:>7.4f}  {vl_f1:>7.4f}")

        if early_stop is not None and early_stop(vl_f1):
            print(f"\n  Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n  Best val Macro F1: {best_val_f1:.4f}")
    return best_val_f1


def extract_rocket_features(X_tr, X_va, X_te, n_kernels=10000):
    try:
        from aeon.transformations.collection.convolution_based import MultiRocket
    except ImportError:
        print("  [WARN] aeon not installed — skipping MultiROCKET.")
        return None, None, None

    print("  Fitting MultiROCKET...")
    Xtr = X_tr.transpose(0, 2, 1).astype(np.float32)
    Xva = X_va.transpose(0, 2, 1).astype(np.float32)
    Xte = X_te.transpose(0, 2, 1).astype(np.float32)

    rocket = MultiRocket(n_kernels=n_kernels)
    Ztr = rocket.fit_transform(Xtr)
    Zva = rocket.transform(Xva)
    Zte = rocket.transform(Xte)
    print(f"  MultiROCKET: {Ztr.shape[1]} dims")
    return Ztr, Zva, Zte


def _pad_time_to_multiple(X_nct, multiple=32):
    N, C, T = X_nct.shape
    if T % multiple == 0:
        return X_nct
    target  = ((T + multiple - 1) // multiple) * multiple
    pad_len = target - T
    pad     = np.repeat(X_nct[:, :, -1:], pad_len, axis=2)
    return np.concatenate([X_nct, pad], axis=2)


def extract_mantis_features(X_tr, X_va, X_te):
    try:
        from mantis.architecture import Mantis8M
        from mantis.trainer import MantisTrainer
    except ImportError:
        print("  [WARN] mantis-tsfm not installed — skipping MANTIS.")
        return None, None, None

    print("  Loading MANTIS-8M...")
    network = Mantis8M(device="cpu")
    network = network.from_pretrained("paris-noah/Mantis-8M")
    model   = MantisTrainer(network=network, device="cpu")

    Xtr = _pad_time_to_multiple(X_tr.transpose(0, 2, 1).astype(np.float32))
    Xva = _pad_time_to_multiple(X_va.transpose(0, 2, 1).astype(np.float32))
    Xte = _pad_time_to_multiple(X_te.transpose(0, 2, 1).astype(np.float32))

    Ztr = model.transform(Xtr)
    Zva = model.transform(Xva)
    Zte = model.transform(Xte)
    print(f"  MANTIS: {Ztr.shape[1]} dims")
    return Ztr, Zva, Zte


# ─────────────────────────────────────────────────────────────────────────────
# LoRA helper — apply peft LoRA to MOMENT backbone
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora_to_moment(moment_model, cfg):
    """
    Apply LoRA adapters to the T5 attention layers inside MOMENTPipeline.
    Returns the total number of trainable LoRA parameters.
    """
    try:
        from peft import get_peft_model, LoraConfig
    except ImportError:
        raise ImportError(
            "peft not installed. Run: pip install peft\n"
            "LoRA requires the peft library from HuggingFace."
        )

    # Disable gradient checkpointing before LoRA (prevents NaN)
    moment_model._disable_gradient_checkpointing()

    # Freeze the entire backbone first
    for p in moment_model.backbone.parameters():
        p.requires_grad = False

    # LoRA config targeting T5 attention Q and V matrices
    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        inference_mode=False,
    )

    # Apply LoRA to the T5 model inside MOMENTPipeline
    # MOMENTPipeline wraps a T5 model at .model attribute
    inner_model = getattr(moment_model.backbone, "model", moment_model.backbone)
    peft_model = get_peft_model(inner_model, lora_config)
    peft_model.print_trainable_parameters()

    # Replace the inner model with the LoRA-wrapped version
    if hasattr(moment_model.backbone, "model"):
        moment_model.backbone.model = peft_model
    else:
        moment_model.backbone = peft_model

    # Unfreeze classification head
    for p in moment_model.head.parameters():
        p.requires_grad = True

    n_lora = sum(p.numel() for n, p in moment_model.backbone.named_parameters()
                 if p.requires_grad and "lora" in n)
    n_head = sum(p.numel() for p in moment_model.head.parameters())
    print(f"  LoRA params: {n_lora:,}  |  Head params: {n_head:,}  |  Total trainable: {n_lora + n_head:,}")
    return n_lora + n_head


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CFG["results_dir"], exist_ok=True)
    seed_everything(CFG["random_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rd     = CFG["results_dir"]

    print_header(f"MOMENT + LoRA (PEFT) + Ensemble  |  Device: {device}")
    print(f"  LoRA rank={CFG['lora_rank']}, alpha={CFG['lora_alpha']}, "
          f"targets={CFG['lora_target_modules']}")

    # ── 1) Load & preprocess ──────────────────────────────────────────────────
    X_train_raw, y_train, X_test_raw, y_test, le = load_lsst()

    strategy    = CFG["imbalance_strategy"]
    use_sampler = strategy in ("both", "sampler_only")
    use_class_w = strategy in ("both", "weights_only")

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
        use_weighted_sampler=use_sampler,
        num_classes=NUM_CLASSES,
    )

    from sklearn.model_selection import StratifiedShuffleSplit
    X_train_norm, _ = preprocess(X_train_raw)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=CFG["val_ratio"],
                                  random_state=CFG["random_state"])
    train_idx, val_idx = next(sss.split(X_train_norm, y_train))
    X_tr_norm = X_train_norm[train_idx]
    X_va_norm = X_train_norm[val_idx]
    y_tr      = y_train[train_idx]
    y_va      = y_train[val_idx]

    criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])

    # ═══════════════════════════════════════════════════════════════════════════
    # Step A — PatchTST
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("Step A: PatchTST")

    patchtst = PatchTSTClassifier(
        seq_len=SEQ_LEN, n_channels=N_CHANNELS, num_classes=NUM_CLASSES,
        patch_len=6, stride=3, d_model=128, n_heads=4, n_layers=4,
        d_ff=256, dropout=0.1, dropout_head=0.2,
    ).to(device)
    print(f"PatchTST: {sum(p.numel() for p in patchtst.parameters()):,} params\n")

    opt_ptst   = torch.optim.AdamW(patchtst.parameters(), lr=CFG["ptst_lr"], weight_decay=1e-4)
    sched_ptst = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ptst, T_max=CFG["ptst_epochs"], eta_min=1e-6)
    es_ptst    = EarlyStopping(patience=CFG["ptst_patience"], mode="max")
    logs_ptst  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

    best_ptst_f1 = train_model(patchtst, train_loader, val_loader, opt_ptst, criterion,
                                sched_ptst, es_ptst, CFG["ptst_epochs"], device, logs_ptst,
                                use_mixup=CFG["use_mixup"], mixup_alpha=CFG["mixup_alpha"])
    torch.save(patchtst.state_dict(), f"{rd}/best_patchtst.pt")
    save_logs(logs_ptst, f"{rd}/logs_patchtst.npz")

    from sklearn.metrics import confusion_matrix as _cm_fn
    _, _, ptst_preds, ptst_targets = eval_epoch(patchtst, test_loader, criterion, device)
    np.save(f"{rd}/patchtst_preds.npy", ptst_preds)
    np.save(f"{rd}/patchtst_targets.npy", ptst_targets)
    np.save(f"{rd}/patchtst_cm.npy", _cm_fn(y_test, ptst_preds))
    a_ptst, f1_ptst, _, _ = compute_metrics(y_test, ptst_preds, le)
    print(f"  PatchTST test — Acc: {a_ptst:.4f}  Macro F1: {f1_ptst:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step B — MOMENT + LoRA fine-tuning
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("Step B: MOMENT + LoRA (PEFT)")

    moment = None
    primary_name = "MOMENT-LoRA"
    logs_moment = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

    try:
        moment = MOMENTClassifier(num_classes=NUM_CLASSES, n_channels=N_CHANNELS, dropout=0.2)
        moment.load_moment(device=str(device))
        moment = moment.to(device)

        # Apply LoRA — replaces linear probing + gradual unfreeze
        print("\n--- Applying LoRA adapters to T5 attention layers ---")
        n_trainable = apply_lora_to_moment(moment, CFG)

        # Build optimizer: differential LR for LoRA vs head
        lora_params = [p for n, p in moment.named_parameters()
                       if p.requires_grad and "lora" in n]
        head_params = list(moment.head.parameters())
        other_trainable = [p for n, p in moment.named_parameters()
                           if p.requires_grad and "lora" not in n
                           and p not in set(head_params)]

        param_groups = [{"params": head_params,  "lr": CFG["lora_head_lr"]},
                        {"params": lora_params,  "lr": CFG["lora_lr"]}]
        if other_trainable:
            param_groups.append({"params": other_trainable, "lr": CFG["lora_lr"] * 0.5})

        opt_lora   = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        sched_lora = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_lora, T_max=CFG["lora_epochs"], eta_min=1e-6)
        es_lora    = EarlyStopping(patience=CFG["lora_patience"], mode="max")

        print(f"\n--- Training LoRA + head ({CFG['lora_epochs']} epochs) ---")
        best_moment_f1 = train_model(
            moment, train_loader, val_loader, opt_lora, criterion,
            sched_lora, es_lora, CFG["lora_epochs"], device, logs_moment,
            use_mixup=CFG["use_mixup"], mixup_alpha=CFG["mixup_alpha"],
        )

        # Save (save_pretrained for peft model if possible, else state_dict)
        try:
            torch.save(moment.state_dict(), f"{rd}/best_moment_lora.pt")
        except Exception:
            torch.save({"head": moment.head.state_dict()}, f"{rd}/best_moment_lora.pt")
        save_logs(logs_moment, f"{rd}/logs_moment.npz")

        primary_f1 = best_moment_f1
        print(f"  MOMENT-LoRA best val Macro F1: {primary_f1:.4f}")

    except Exception as e:
        print(f"\n  [WARN] MOMENT-LoRA failed: {e}")
        print("  Falling back to InceptionTime-Large.\n")
        import traceback; traceback.print_exc()
        moment = None
        primary_name = "InceptionTime-Large (fallback)"

    if moment is None:
        print_header("Step B fallback: InceptionTime-Large")
        moment = InceptionTime(
            n_channels=N_CHANNELS, num_classes=NUM_CLASSES,
            nb_filters=64, kernel_sizes=(9, 19, 39), n_blocks=3, dropout=0.2,
        ).to(device)
        opt_fb   = torch.optim.AdamW(moment.parameters(), lr=1e-3, weight_decay=1e-4)
        sched_fb = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fb, T_max=200, eta_min=1e-6)
        es_fb    = EarlyStopping(patience=25, mode="max")
        primary_f1 = train_model(moment, train_loader, val_loader, opt_fb, criterion,
                                  sched_fb, es_fb, 200, device, logs_moment,
                                  use_mixup=CFG["use_mixup"], mixup_alpha=CFG["mixup_alpha"])
        torch.save(moment.state_dict(), f"{rd}/best_moment.pt")
        save_logs(logs_moment, f"{rd}/logs_moment.npz")

    # Save primary predictions + embeddings
    _, _, prim_preds, prim_targets = eval_epoch(moment, test_loader, criterion, device)
    np.save(f"{rd}/moment_preds.npy", prim_preds)
    np.save(f"{rd}/moment_targets.npy", prim_targets)
    np.save(f"{rd}/moment_cm.npy", _cm_fn(y_test, prim_preds))
    a_prim, f1_prim, _, _ = compute_metrics(y_test, prim_preds, le)
    print(f"  {primary_name} test — Acc: {a_prim:.4f}  Macro F1: {f1_prim:.4f}")

    try:
        emb_te, lbl_te = get_embeddings(moment, test_loader, device)
        np.save(f"{rd}/moment_embeddings.npy", emb_te)
        np.save(f"{rd}/moment_emb_labels.npy", lbl_te)
    except Exception as e:
        print(f"  [WARN] Embeddings: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step B2 — InceptionTime-Large (always trained)
    # ═══════════════════════════════════════════════════════════════════════════
    inception_large = None
    va_f1_inception = 0.0
    if CFG.get("use_inception_large", True):
        print_header("Step B2: InceptionTime-Large (additional ensemble member)")
        inception_large = InceptionTime(
            n_channels=N_CHANNELS, num_classes=NUM_CLASSES,
            nb_filters=64, kernel_sizes=(9, 19, 39), n_blocks=3, dropout=0.2,
        ).to(device)
        print(f"InceptionTime-Large: {sum(p.numel() for p in inception_large.parameters()):,} params\n")
        logs_inc  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
        opt_inc   = torch.optim.AdamW(inception_large.parameters(), lr=1e-3, weight_decay=1e-4)
        sched_inc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_inc, T_max=CFG["inception_epochs"], eta_min=1e-6)
        es_inc    = EarlyStopping(patience=25, mode="max")
        va_f1_inception = train_model(inception_large, train_loader, val_loader, opt_inc, criterion,
                                      sched_inc, es_inc, CFG["inception_epochs"], device, logs_inc,
                                      use_mixup=CFG["use_mixup"], mixup_alpha=CFG["mixup_alpha"])
        torch.save(inception_large.state_dict(), f"{rd}/best_inception_large.pt")
        _, _, inc_preds, _ = eval_epoch(inception_large, test_loader, criterion, device)
        a_inc, f1_inc, _, _ = compute_metrics(y_test, inc_preds, le)
        print(f"  InceptionTime-Large test — Acc: {a_inc:.4f}  Macro F1: {f1_inc:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Steps C/D — Feature extraction
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("Steps C/D: Feature extraction")

    Z_mr_tr, Z_mr_va, Z_mr_te = extract_rocket_features(
        X_tr_norm, X_va_norm, X_test_norm, n_kernels=CFG["rocket_n_kernels"])

    Z_ma_tr = Z_ma_va = Z_ma_te = None
    try:
        Z_ma_tr, Z_ma_va, Z_ma_te = extract_mantis_features(X_tr_norm, X_va_norm, X_test_norm)
    except Exception as e:
        print(f"  [WARN] MANTIS: {e}")

    print("  Extracting MOMENT-LoRA embeddings...")
    Z_pm_tr, _ = get_embeddings(moment, train_loader, device)
    Z_pm_va, _ = get_embeddings(moment, val_loader,   device)
    Z_pm_te, _ = get_embeddings(moment, test_loader,  device)
    print(f"  MOMENT embedding dim: {Z_pm_tr.shape[1]}")

    feat_parts_tr = [Z_pm_tr]
    feat_parts_va = [Z_pm_va]
    feat_parts_te = [Z_pm_te]

    if Z_mr_tr is not None:
        feat_parts_tr.append(Z_mr_tr); feat_parts_va.append(Z_mr_va); feat_parts_te.append(Z_mr_te)
    if Z_ma_tr is not None:
        feat_parts_tr.append(Z_ma_tr); feat_parts_va.append(Z_ma_va); feat_parts_te.append(Z_ma_te)

    feat_tr = np.concatenate(feat_parts_tr, axis=1).astype(np.float32)
    feat_va = np.concatenate(feat_parts_va, axis=1).astype(np.float32)
    feat_te = np.concatenate(feat_parts_te, axis=1).astype(np.float32)
    print(f"  Total feature dim: {feat_tr.shape[1]}")

    from sklearn.preprocessing import StandardScaler
    sk_scaler = StandardScaler().fit(feat_tr)
    feat_tr_s = sk_scaler.transform(feat_tr).astype(np.float32)
    feat_va_s = sk_scaler.transform(feat_va).astype(np.float32)
    feat_te_s = sk_scaler.transform(feat_te).astype(np.float32)

    # Ridge(ROCKET)
    ridge_clf = None
    va_f1_ridge = 0.0
    ridge_va_probs = ridge_te_probs = None
    if Z_mr_tr is not None:
        print("\n  Training Ridge(ROCKET)...")
        from sklearn.linear_model import RidgeClassifierCV
        from sklearn.preprocessing import StandardScaler as MRScaler
        mr_sc = MRScaler().fit(Z_mr_tr)
        Z_mr_tr_s = mr_sc.transform(Z_mr_tr)
        Z_mr_va_s = mr_sc.transform(Z_mr_va)
        Z_mr_te_s = mr_sc.transform(Z_mr_te)
        ridge_clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced")
        ridge_clf.fit(Z_mr_tr_s, y_tr)
        ridge_va_preds = ridge_clf.predict(Z_mr_va_s)
        va_f1_ridge = f1_score(y_va, ridge_va_preds, average="macro", zero_division=0)
        print(f"  Ridge(ROCKET) val Macro F1: {va_f1_ridge:.4f}")

        def _softmax(x):
            e = np.exp(x - x.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        ridge_va_probs = _softmax(ridge_clf.decision_function(Z_mr_va_s))
        ridge_te_probs = _softmax(ridge_clf.decision_function(Z_mr_te_s))

    # ═══════════════════════════════════════════════════════════════════════════
    # Step E — Feature-stacking MLP
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("Step E: Feature-stacking MLP")

    from torch.utils.data import TensorDataset, DataLoader as DL

    def make_feat_loader(F, y, weighted=False, bs=64):
        ds = TensorDataset(
            torch.tensor(F, dtype=torch.float32),
            torch.zeros(len(F), dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        if weighted:
            sampler = get_weighted_sampler(y, num_classes=NUM_CLASSES)
            return DL(ds, batch_size=bs, sampler=sampler)
        return DL(ds, batch_size=bs, shuffle=True)

    feat_train_loader = make_feat_loader(feat_tr_s, y_tr,    weighted=True,  bs=64)
    feat_val_loader   = make_feat_loader(feat_va_s, y_va,    weighted=False, bs=64)
    feat_test_loader  = make_feat_loader(feat_te_s, y_test,  weighted=False, bs=64)

    in_dim   = feat_tr.shape[1]
    feat_mlp = FeatureMLP(in_dim=in_dim, hidden=CFG["mlp_hidden"],
                          num_classes=NUM_CLASSES, dropout=CFG["mlp_dropout"]).to(device)
    print(f"FeatureMLP: {sum(p.numel() for p in feat_mlp.parameters()):,} params\n")

    mlp_class_w   = compute_class_weights(y_tr, num_classes=NUM_CLASSES).to(device)
    criterion_mlp = nn.CrossEntropyLoss(weight=mlp_class_w, label_smoothing=CFG["label_smoothing"])

    opt_mlp   = torch.optim.AdamW(feat_mlp.parameters(), lr=CFG["mlp_lr"], weight_decay=1e-4)
    sched_mlp = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_mlp, mode="max", factor=0.5, patience=7, min_lr=1e-6)
    es_mlp    = EarlyStopping(patience=CFG["mlp_patience"], mode="max")
    logs_mlp  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

    best_mlp_f1 = train_model(feat_mlp, feat_train_loader, feat_val_loader,
                               opt_mlp, criterion_mlp, sched_mlp, es_mlp,
                               CFG["mlp_epochs"], device, logs_mlp, use_mixup=False)
    torch.save(feat_mlp.state_dict(), f"{rd}/best_feature_mlp.pt")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step F — Soft-voting ensemble
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("Step F: Soft-voting ensemble")

    _, _, prim_va_preds, prim_va_targets = eval_epoch(moment, val_loader, criterion, device)
    va_f1_prim = f1_score(prim_va_targets, prim_va_preds, average="macro", zero_division=0)

    _, _, ptst_va_preds, ptst_va_targets = eval_epoch(patchtst, val_loader, criterion, device)
    va_f1_ptst = f1_score(ptst_va_targets, ptst_va_preds, average="macro", zero_division=0)

    _, _, mlp_va_preds, mlp_va_targets = eval_epoch(feat_mlp, feat_val_loader, criterion_mlp, device)
    va_f1_mlp = f1_score(mlp_va_targets, mlp_va_preds, average="macro", zero_division=0)

    if inception_large is not None:
        _, _, inc_va_preds, inc_va_targets = eval_epoch(inception_large, val_loader, criterion, device)
        va_f1_inception = f1_score(inc_va_targets, inc_va_preds, average="macro", zero_division=0)

    print(f"  Val F1 — MOMENT-LoRA:         {va_f1_prim:.4f}")
    print(f"  Val F1 — InceptionTime-Large: {va_f1_inception:.4f}")
    print(f"  Val F1 — PatchTST:            {va_f1_ptst:.4f}")
    print(f"  Val F1 — FeatureMLP:          {va_f1_mlp:.4f}")
    print(f"  Val F1 — Ridge(ROCKET):       {va_f1_ridge:.4f}")

    f1s     = np.array([va_f1_prim, va_f1_inception, va_f1_ptst, va_f1_mlp, va_f1_ridge])
    weights = np.exp(f1s * 10)
    weights /= weights.sum()
    print(f"  Ensemble weights: {np.round(weights, 4)}")

    @torch.no_grad()
    def get_probs(model, loader, dev):
        model.eval()
        out = []
        for x, mask, y in loader:
            x, mask = x.to(dev), mask.to(dev)
            out.append(torch.softmax(model(x, mask), dim=-1).cpu().numpy())
        return np.vstack(out)

    probs_prim = np.nan_to_num(get_probs(moment,         test_loader,      device), nan=1.0/NUM_CLASSES)
    probs_inc  = np.nan_to_num(get_probs(inception_large, test_loader,     device), nan=1.0/NUM_CLASSES) if inception_large is not None else np.full((len(y_test), NUM_CLASSES), 1.0/NUM_CLASSES)
    probs_ptst = np.nan_to_num(get_probs(patchtst,        test_loader,     device), nan=1.0/NUM_CLASSES)
    probs_mlp  = np.nan_to_num(get_probs(feat_mlp,        feat_test_loader, device), nan=1.0/NUM_CLASSES)
    probs_ridg = np.nan_to_num(ridge_te_probs, nan=1.0/NUM_CLASSES) if ridge_te_probs is not None else np.full((len(y_test), NUM_CLASSES), 1.0/NUM_CLASSES)

    probs_ens = (weights[0]*probs_prim + weights[1]*probs_inc + weights[2]*probs_ptst
                 + weights[3]*probs_mlp + weights[4]*probs_ridg)
    preds_ens = probs_ens.argmax(axis=1)

    acc_ens, f1_ens, report_ens, cm_ens = compute_metrics(y_test, preds_ens, le)

    print(f"\n{'='*60}")
    print(f"  === FINAL ENSEMBLE (MOMENT-LoRA) ===")
    print(f"  Accuracy : {acc_ens:.4f}")
    print(f"  Macro F1 : {f1_ens:.4f}")
    print(f"{'='*60}")
    print(report_ens)

    np.save(f"{rd}/competitor_preds.npy",   preds_ens)
    np.save(f"{rd}/competitor_probs.npy",   probs_ens)
    np.save(f"{rd}/competitor_targets.npy", y_test)
    np.save(f"{rd}/competitor_cm.npy",      cm_ens)

    with open(f"{rd}/results_lora.txt", "w") as f:
        f.write(f"MOMENT-LoRA Ensemble Summary\n{'='*40}\n\n")
        f.write(f"LoRA rank={CFG['lora_rank']}, alpha={CFG['lora_alpha']}, "
                f"targets={CFG['lora_target_modules']}\n\n")
        f.write(f"MOMENT-LoRA:         Acc={a_prim:.4f}  F1={f1_prim:.4f}\n")
        f.write(f"InceptionTime-Large: Acc={a_inc:.4f}  F1={f1_inc:.4f}\n" if inception_large else "")
        f.write(f"PatchTST:            Acc={a_ptst:.4f}  F1={f1_ptst:.4f}\n")
        f.write(f"Ensemble:            Acc={acc_ens:.4f}  F1={f1_ens:.4f}\n\n")
        f.write(f"Classification Report:\n{report_ens}\n")

    print(f"\n  All outputs saved to {rd}/")
    print(f"  Run evaluate.py to generate all figures.\n")
    return acc_ens, f1_ens


if __name__ == "__main__":
    main()
