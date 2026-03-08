"""
train_super_competitor.py — Super-Competitor pipeline.

Models (6 total):
  1. MOMENT-1-large  (foundation model, 3-phase fine-tune)
  2. ConvTran        (Convolution+Transformer, SOTA on UEA)
  3. InceptionTime-XL (nb_filters=128, n_blocks=4)
  4. PatchTST-Large  (d_model=256, n_heads=8)
  5. MultiROCKET + XGBoost
  6. FeatureMLP      (embedding stacking)

Ensemble:
  - Soft-voting weighted by exp(val_F1 * 10)
  - Stacking meta-learner: LogReg on 6 × 14 = 84 softmax features
  - Test-Time Augmentation (TTA, 5 passes)
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
    compute_class_weights, get_weighted_sampler,
)
from models.moment_classifier import MOMENTClassifier, PatchTSTClassifier
from models.inception_time import InceptionTime
from models.convtran import ConvTran
from utils import (
    seed_everything, EarlyStopping, train_epoch, eval_epoch,
    compute_metrics, get_embeddings, save_logs,
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

    # MOMENT fine-tuning
    "use_moment":           True,
    "moment_phases":        3,          # 1=linear probe, 2=+unfreeze, 3=+full ft
    "lp_epochs":            50,
    "lp_lr":                1e-3,
    "gu_epochs":            40,
    "gu_n_blocks":          2,
    "gu_lr_head":           5e-4,
    "gu_lr_enc":            5e-7,       # ultra-low — prevents NaN in T5 layers
    "ft_epochs":            30,
    "ft_lr_head":           1e-4,
    "ft_lr_enc":            1e-7,
    "ft_patience":          10,

    # ConvTran
    "convtran_epochs":      200,
    "convtran_lr":          5e-4,
    "convtran_patience":    25,

    # InceptionTime-XL
    "inception_epochs":     200,
    "inception_lr":         1e-3,
    "inception_patience":   25,

    # PatchTST-Large
    "ptst_epochs":          150,
    "ptst_lr":              5e-4,
    "ptst_patience":        20,

    # MultiROCKET
    "rocket_n_kernels":     10000,

    # Feature MLP
    "mlp_hidden":           256,
    "mlp_dropout":          0.3,
    "mlp_epochs":           100,
    "mlp_lr":               1e-3,
    "mlp_patience":         15,

    # Stacking meta-learner
    "stacking_C":           1.0,

    # Test-Time Augmentation
    "tta_passes":           5,

    "results_dir":          "results",
}

NUM_CLASSES = 14
N_CHANNELS  = 6
SEQ_LEN     = 36


# ─────────────────────────────────────────────────────────────────────────────
# Feature MLP
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_header(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}\n")


def log_param_groups(optimizer, label=""):
    tag = f" [{label}]" if label else ""
    for i, pg in enumerate(optimizer.param_groups):
        n = sum(p.numel() for p in pg["params"] if p.requires_grad)
        print(f"    opt group {i}{tag}: lr={pg['lr']:.2e}, trainable={n:,} params")


def train_model(model, train_loader, val_loader, optimizer, criterion,
                scheduler, early_stop, n_epochs, device, logs,
                use_mixup=False, mixup_alpha=0.3):
    best_f1    = 0.0
    best_state = None

    hdr = (f"{'Epoch':>6}  {'TrLoss':>8}  {'TrAcc':>7}  "
           f"{'VLoss':>8}  {'VAcc':>7}  {'VF1':>7}")
    print(hdr)
    print("-" * len(hdr))

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_mixup=use_mixup, mixup_alpha=mixup_alpha,
        )
        vl_loss, vl_acc, vl_preds, vl_tgts = eval_epoch(
            model, val_loader, criterion, device,
        )
        vl_f1 = f1_score(vl_tgts, vl_preds, average="macro", zero_division=0)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(vl_f1)
            else:
                scheduler.step()

        for key, val in [("train_loss", tr_loss), ("val_loss", vl_loss),
                         ("train_acc", tr_acc),   ("val_acc",  vl_acc)]:
            if key in logs:
                logs[key].append(val)
        if "val_f1" in logs:
            logs["val_f1"].append(vl_f1)

        if vl_f1 > best_f1:
            best_f1    = vl_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"{epoch:>6d}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  "
              f"{vl_loss:>8.4f}  {vl_acc:>7.4f}  {vl_f1:>7.4f}")

        if early_stop is not None and early_stop(vl_f1):
            print(f"\n  Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n  Best val Macro F1: {best_f1:.4f}")
    return best_f1


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_rocket_features(X_tr, X_va, X_te, n_kernels=10000):
    try:
        from aeon.transformations.collection.convolution_based import MultiRocket
    except ImportError:
        print("  [WARN] aeon not installed — skipping MultiROCKET.")
        return None, None, None
    print("  Fitting MultiROCKET...")
    def _T(X): return X.transpose(0, 2, 1).astype(np.float32)
    rocket = MultiRocket(n_kernels=n_kernels)
    Ztr = rocket.fit_transform(_T(X_tr))
    Zva = rocket.transform(_T(X_va))
    Zte = rocket.transform(_T(X_te))
    print(f"  MultiROCKET: {Ztr.shape[1]} dims")
    return Ztr, Zva, Zte


def _pad_time_to_multiple(X_nct, multiple=32):
    N, C, T = X_nct.shape
    if T % multiple == 0:
        return X_nct
    target  = ((T + multiple - 1) // multiple) * multiple
    pad     = np.repeat(X_nct[:, :, -1:], target - T, axis=2)
    return np.concatenate([X_nct, pad], axis=2)


def extract_mantis_features(X_tr, X_va, X_te):
    try:
        from mantis.architecture import Mantis8M
        from mantis.trainer import MantisTrainer
    except ImportError:
        print("  [WARN] mantis-tsfm not installed — skipping MANTIS.")
        return None, None, None
    print("  Loading MANTIS-8M...")
    network = Mantis8M(device="cpu").from_pretrained("paris-noah/Mantis-8M")
    model   = MantisTrainer(network=network, device="cpu")
    def _prep(X): return _pad_time_to_multiple(X.transpose(0,2,1).astype(np.float32))
    Ztr = model.transform(_prep(X_tr))
    Zva = model.transform(_prep(X_va))
    Zte = model.transform(_prep(X_te))
    print(f"  MANTIS: {Ztr.shape[1]} dims")
    return Ztr, Zva, Zte


# ─────────────────────────────────────────────────────────────────────────────
# Soft probabilities (with nan guard)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_probs(model, loader, device):
    model.eval()
    out = []
    for x, mask, _ in loader:
        x, mask = x.to(device), mask.to(device)
        p = torch.softmax(model(x, mask), dim=-1).cpu().numpy()
        out.append(p)
    probs = np.vstack(out)
    return np.nan_to_num(probs, nan=1.0 / NUM_CLASSES)


# ─────────────────────────────────────────────────────────────────────────────
# Test-Time Augmentation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_probs_tta(model, loader, device, n_passes=5, jitter_sigma=0.03):
    """Average softmax over n_passes jitter augmentations."""
    model.eval()
    accumulated = None
    for _ in range(n_passes):
        out = []
        for x, mask, _ in loader:
            x, mask = x.to(device), mask.to(device)
            x_aug = x + torch.randn_like(x) * jitter_sigma
            p = torch.softmax(model(x_aug, mask), dim=-1).cpu().numpy()
            out.append(p)
        probs = np.nan_to_num(np.vstack(out), nan=1.0 / NUM_CLASSES)
        accumulated = probs if accumulated is None else accumulated + probs
    return accumulated / n_passes


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CFG["results_dir"], exist_ok=True)
    seed_everything(CFG["random_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rd     = CFG["results_dir"]

    print_header(f"SUPER COMPETITOR — 6 models + Stacking + TTA  |  Device: {device}")

    # ── 1) Load ───────────────────────────────────────────────────────────────
    X_train_raw, y_train, X_test_raw, y_test, le = load_lsst()

    use_sampler = CFG["imbalance_strategy"] in ("both", "sampler_only")
    use_class_w = CFG["imbalance_strategy"] in ("both", "weights_only")

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

    # Raw arrays for feature extraction
    from sklearn.model_selection import StratifiedShuffleSplit
    X_train_norm, _ = preprocess(X_train_raw)
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=CFG["val_ratio"], random_state=CFG["random_state"],
    )
    train_idx, val_idx = next(sss.split(X_train_norm, y_train))
    X_tr_norm = X_train_norm[train_idx]
    X_va_norm = X_train_norm[val_idx]
    y_tr      = y_train[train_idx]
    y_va      = y_train[val_idx]

    # ── 2) Loss ───────────────────────────────────────────────────────────────
    if use_class_w:
        class_w = compute_class_weights(y_train, num_classes=NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=CFG["label_smoothing"])
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])

    from sklearn.metrics import confusion_matrix as _cm_fn

    # ═══════════════════════════════════════════════════════════════════════════
    # Step A — MOMENT foundation model (3-phase fine-tuning)
    # ═══════════════════════════════════════════════════════════════════════════
    moment_model = None
    va_f1_moment = 0.0

    if CFG["use_moment"]:
        try:
            import momentfm  # noqa
            print_header("Step A: MOMENT-1-large fine-tuning (3 phases)")
            moment_model = MOMENTClassifier(
                num_classes=NUM_CLASSES, n_channels=N_CHANNELS, dropout=0.2,
            )
            moment_model.load_moment(device=str(device))
            moment_model = moment_model.to(device)
            logs_moment = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

            # — Phase 1: linear probing ————————————————————————————————————————
            print(f"\n--- Phase 1: Linear probing ({CFG['lp_epochs']} epochs) ---")
            moment_model.freeze_encoder()
            opt_lp = torch.optim.AdamW(
                [p for p in moment_model.parameters() if p.requires_grad],
                lr=CFG["lp_lr"], weight_decay=1e-4,
            )
            sched_lp = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_lp, T_max=CFG["lp_epochs"], eta_min=1e-6,
            )
            va_f1_moment = train_model(
                moment_model, train_loader, val_loader, opt_lp, criterion,
                sched_lp, EarlyStopping(patience=15, mode="max"),
                CFG["lp_epochs"], device, logs_moment, use_mixup=False,
            )

            # — Phase 2: gradual unfreeze ——————————————————————————————————————
            if CFG["moment_phases"] >= 2:
                print(f"\n--- Phase 2: Gradual unfreeze ({CFG['gu_epochs']} epochs) ---")
                moment_model.unfreeze_last_n(n=CFG["gu_n_blocks"])
                param_groups = moment_model.get_param_groups(
                    lr_head=CFG["gu_lr_head"], lr_encoder=CFG["gu_lr_enc"],
                )
                opt_gu = torch.optim.AdamW(param_groups, weight_decay=1e-4)
                log_param_groups(opt_gu, "Phase 2")
                sched_gu = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt_gu, T_max=CFG["gu_epochs"], eta_min=1e-7,
                )
                va_f1_moment = train_model(
                    moment_model, train_loader, val_loader, opt_gu, criterion,
                    sched_gu, EarlyStopping(patience=10, mode="max"),
                    CFG["gu_epochs"], device, logs_moment,
                    use_mixup=CFG["use_mixup"], mixup_alpha=CFG["mixup_alpha"],
                )

            # — Phase 3: full fine-tune ————————————————————————————————————————
            if CFG["moment_phases"] >= 3 and CFG["ft_epochs"] > 0:
                print(f"\n--- Phase 3: Full fine-tuning ({CFG['ft_epochs']} epochs) ---")
                moment_model.unfreeze_all()
                param_groups = moment_model.get_param_groups(
                    lr_head=CFG["ft_lr_head"], lr_encoder=CFG["ft_lr_enc"],
                )
                opt_ft = torch.optim.AdamW(param_groups, weight_decay=1e-2)
                log_param_groups(opt_ft, "Phase 3")
                sched_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt_ft, T_max=CFG["ft_epochs"], eta_min=1e-8,
                )
                va_f1_moment = train_model(
                    moment_model, train_loader, val_loader, opt_ft, criterion,
                    sched_ft, EarlyStopping(patience=CFG["ft_patience"], mode="max"),
                    CFG["ft_epochs"], device, logs_moment,
                    use_mixup=CFG["use_mixup"], mixup_alpha=CFG["mixup_alpha"],
                )

            torch.save(moment_model.state_dict(), f"{rd}/best_moment.pt")
            save_logs(logs_moment, f"{rd}/logs_moment.npz")

            # immediate save of predictions
            _, _, prim_preds, _ = eval_epoch(moment_model, test_loader, criterion, device)
            np.save(f"{rd}/moment_preds.npy", prim_preds)
            np.save(f"{rd}/moment_targets.npy", y_test)
            np.save(f"{rd}/moment_cm.npy", _cm_fn(y_test, prim_preds))
            a_m, f1_m, _, _ = compute_metrics(y_test, prim_preds, le)
            print(f"\n  MOMENT test — Acc: {a_m:.4f}  Macro F1: {f1_m:.4f}")

            try:
                emb_te, lbl_te = get_embeddings(moment_model, test_loader, device)
                np.save(f"{rd}/moment_embeddings.npy", emb_te)
                np.save(f"{rd}/moment_emb_labels.npy", lbl_te)
            except Exception as e:
                print(f"  [WARN] Embedding extraction: {e}")

        except Exception as err:
            print(f"\n  [WARN] MOMENT failed: {err} — skipping MOMENT.\n")
            moment_model = None

    # ═══════════════════════════════════════════════════════════════════════════
    # Step B — ConvTran
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("Step B: ConvTran (Conv+Transformer)")
    convtran = ConvTran(
        n_channels=N_CHANNELS, seq_len=SEQ_LEN, num_classes=NUM_CLASSES,
        d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.1,
    ).to(device)
    print(f"ConvTran: {sum(p.numel() for p in convtran.parameters()):,} params\n")

    logs_ct = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    opt_ct   = torch.optim.AdamW(convtran.parameters(), lr=CFG["convtran_lr"], weight_decay=1e-4)
    sched_ct = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_ct, T_max=CFG["convtran_epochs"], eta_min=1e-6,
    )
    va_f1_ct = train_model(
        convtran, train_loader, val_loader, opt_ct, criterion,
        sched_ct, EarlyStopping(patience=CFG["convtran_patience"], mode="max"),
        CFG["convtran_epochs"], device, logs_ct,
        use_mixup=CFG["use_mixup"], mixup_alpha=CFG["mixup_alpha"],
    )
    torch.save(convtran.state_dict(), f"{rd}/best_convtran.pt")
    save_logs(logs_ct, f"{rd}/logs_convtran.npz")

    _, _, ct_preds, _ = eval_epoch(convtran, test_loader, criterion, device)
    a_ct, f1_ct, _, _ = compute_metrics(y_test, ct_preds, le)
    print(f"  ConvTran test — Acc: {a_ct:.4f}  Macro F1: {f1_ct:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step C — InceptionTime-XL
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("Step C: InceptionTime-XL (nb_filters=128, n_blocks=4)")
    inception_xl = InceptionTime(
        n_channels=N_CHANNELS, num_classes=NUM_CLASSES,
        nb_filters=128, kernel_sizes=(9, 19, 39), n_blocks=4, dropout=0.2,
    ).to(device)
    print(f"InceptionTime-XL: {sum(p.numel() for p in inception_xl.parameters()):,} params\n")

    logs_inc = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    opt_inc   = torch.optim.AdamW(inception_xl.parameters(), lr=CFG["inception_lr"], weight_decay=1e-4)
    sched_inc = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_inc, T_max=CFG["inception_epochs"], eta_min=1e-6,
    )
    va_f1_inc = train_model(
        inception_xl, train_loader, val_loader, opt_inc, criterion,
        sched_inc, EarlyStopping(patience=CFG["inception_patience"], mode="max"),
        CFG["inception_epochs"], device, logs_inc,
        use_mixup=CFG["use_mixup"], mixup_alpha=CFG["mixup_alpha"],
    )
    torch.save(inception_xl.state_dict(), f"{rd}/best_inception_xl.pt")
    save_logs(logs_inc, f"{rd}/logs_inception_xl.npz")

    _, _, inc_preds, _ = eval_epoch(inception_xl, test_loader, criterion, device)
    a_inc, f1_inc, _, _ = compute_metrics(y_test, inc_preds, le)
    print(f"  InceptionTime-XL test — Acc: {a_inc:.4f}  Macro F1: {f1_inc:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step D — PatchTST-Large
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("Step D: PatchTST-Large (d_model=256, n_heads=8)")
    patchtst = PatchTSTClassifier(
        seq_len=SEQ_LEN, n_channels=N_CHANNELS, num_classes=NUM_CLASSES,
        patch_len=6, stride=3, d_model=256, n_heads=8, n_layers=6,
        d_ff=512, dropout=0.1, dropout_head=0.2,
    ).to(device)
    print(f"PatchTST-Large: {sum(p.numel() for p in patchtst.parameters()):,} params\n")

    logs_ptst  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    opt_ptst   = torch.optim.AdamW(patchtst.parameters(), lr=CFG["ptst_lr"], weight_decay=1e-4)
    sched_ptst = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_ptst, T_max=CFG["ptst_epochs"], eta_min=1e-6,
    )
    va_f1_ptst = train_model(
        patchtst, train_loader, val_loader, opt_ptst, criterion,
        sched_ptst, EarlyStopping(patience=CFG["ptst_patience"], mode="max"),
        CFG["ptst_epochs"], device, logs_ptst,
        use_mixup=CFG["use_mixup"], mixup_alpha=CFG["mixup_alpha"],
    )
    torch.save(patchtst.state_dict(), f"{rd}/best_patchtst.pt")
    save_logs(logs_ptst, f"{rd}/logs_patchtst.npz")

    _, _, ptst_preds, _ = eval_epoch(patchtst, test_loader, criterion, device)
    a_ptst, f1_ptst, _, _ = compute_metrics(y_test, ptst_preds, le)
    print(f"  PatchTST-Large test — Acc: {a_ptst:.4f}  Macro F1: {f1_ptst:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step E — Feature extraction + XGBoost + FeatureMLP
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("Step E: Feature extraction (MultiROCKET + MANTIS)")

    Z_mr_tr, Z_mr_va, Z_mr_te = extract_rocket_features(
        X_tr_norm, X_va_norm, X_test_norm, n_kernels=CFG["rocket_n_kernels"],
    )
    Z_ma_tr = Z_ma_va = Z_ma_te = None
    try:
        Z_ma_tr, Z_ma_va, Z_ma_te = extract_mantis_features(X_tr_norm, X_va_norm, X_test_norm)
    except Exception as e:
        print(f"  [WARN] MANTIS: {e}")

    # Embeddings from best scratch model (InceptionTime-XL or ConvTran)
    print("  Extracting ConvTran embeddings for FeatureMLP...")
    Z_emb_tr, _ = get_embeddings(convtran, train_loader, device)
    Z_emb_va, _ = get_embeddings(convtran, val_loader,   device)
    Z_emb_te, _ = get_embeddings(convtran, test_loader,  device)

    feat_parts_tr = [Z_emb_tr]
    feat_parts_va = [Z_emb_va]
    feat_parts_te = [Z_emb_te]

    if Z_mr_tr is not None:
        feat_parts_tr.append(Z_mr_tr)
        feat_parts_va.append(Z_mr_va)
        feat_parts_te.append(Z_mr_te)
    if Z_ma_tr is not None:
        feat_parts_tr.append(Z_ma_tr)
        feat_parts_va.append(Z_ma_va)
        feat_parts_te.append(Z_ma_te)

    feat_tr = np.concatenate(feat_parts_tr, axis=1).astype(np.float32)
    feat_va = np.concatenate(feat_parts_va, axis=1).astype(np.float32)
    feat_te = np.concatenate(feat_parts_te, axis=1).astype(np.float32)
    print(f"  Total feature dim: {feat_tr.shape[1]}")

    from sklearn.preprocessing import StandardScaler
    sk_sc = StandardScaler().fit(feat_tr)
    feat_tr_s = sk_sc.transform(feat_tr).astype(np.float32)
    feat_va_s = sk_sc.transform(feat_va).astype(np.float32)
    feat_te_s = sk_sc.transform(feat_te).astype(np.float32)

    # — XGBoost on MultiROCKET features ————————————————————————————————————————
    xgb_model   = None
    va_f1_xgb   = 0.0
    xgb_va_probs = xgb_te_probs = None

    if Z_mr_tr is not None:
        try:
            import xgboost as xgb
            print("\n  Training XGBoost(ROCKET)...")
            from sklearn.preprocessing import StandardScaler as MRSc
            mr_sc = MRSc().fit(Z_mr_tr)
            Ztr_s = mr_sc.transform(Z_mr_tr)
            Zva_s = mr_sc.transform(Z_mr_va)
            Zte_s = mr_sc.transform(Z_mr_te)

            # compute class frequencies for scale_pos_weight equivalent
            xgb_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.6,
                use_label_encoder=False,
                eval_metric="mlogloss",
                num_class=NUM_CLASSES,
                tree_method="hist",
                device="cuda" if torch.cuda.is_available() else "cpu",
                random_state=42,
                verbosity=0,
            )
            xgb_model.fit(Ztr_s, y_tr, eval_set=[(Zva_s, y_va)], verbose=False)
            xgb_va_probs  = xgb_model.predict_proba(Zva_s)
            xgb_te_probs  = xgb_model.predict_proba(Zte_s)
            va_f1_xgb = f1_score(y_va, xgb_va_probs.argmax(1), average="macro", zero_division=0)
            print(f"  XGBoost(ROCKET) val Macro F1: {va_f1_xgb:.4f}")

        except ImportError:
            print("  [WARN] xgboost not installed — using Ridge fallback.")
            from sklearn.linear_model import RidgeClassifierCV
            from sklearn.preprocessing import StandardScaler as MRSc
            mr_sc = MRSc().fit(Z_mr_tr)
            Ztr_s = mr_sc.transform(Z_mr_tr)
            Zva_s = mr_sc.transform(Z_mr_va)
            Zte_s = mr_sc.transform(Z_mr_te)
            ridge = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced")
            ridge.fit(Ztr_s, y_tr)
            def _softmax(x):
                e = np.exp(x - x.max(axis=1, keepdims=True))
                return e / e.sum(axis=1, keepdims=True)
            xgb_va_probs = _softmax(ridge.decision_function(Zva_s))
            xgb_te_probs = _softmax(ridge.decision_function(Zte_s))
            va_f1_xgb = f1_score(y_va, ridge.predict(Zva_s), average="macro", zero_division=0)
            print(f"  Ridge(ROCKET) val Macro F1: {va_f1_xgb:.4f}")

    # — FeatureMLP ——————————————————————————————————————————————————————————————
    print_header("Step E2: FeatureMLP")
    from torch.utils.data import TensorDataset, DataLoader as DL

    def make_feat_loader(F, y, weighted=False, bs=64):
        ds = TensorDataset(
            torch.tensor(F, dtype=torch.float32),
            torch.zeros(len(F), dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        if weighted:
            return DL(ds, batch_size=bs, sampler=get_weighted_sampler(y, num_classes=NUM_CLASSES))
        return DL(ds, batch_size=bs, shuffle=True)

    feat_train_ldr = make_feat_loader(feat_tr_s, y_tr,   weighted=True)
    feat_val_ldr   = make_feat_loader(feat_va_s, y_va,   weighted=False)
    feat_test_ldr  = make_feat_loader(feat_te_s, y_test, weighted=False)

    feat_mlp = FeatureMLP(
        in_dim=feat_tr.shape[1], hidden=CFG["mlp_hidden"],
        num_classes=NUM_CLASSES, dropout=CFG["mlp_dropout"],
    ).to(device)
    print(f"FeatureMLP: {sum(p.numel() for p in feat_mlp.parameters()):,} params\n")

    mlp_class_w  = compute_class_weights(y_tr, num_classes=NUM_CLASSES).to(device)
    criterion_mlp = nn.CrossEntropyLoss(weight=mlp_class_w, label_smoothing=CFG["label_smoothing"])
    opt_mlp   = torch.optim.AdamW(feat_mlp.parameters(), lr=CFG["mlp_lr"], weight_decay=1e-4)
    sched_mlp = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_mlp, mode="max", factor=0.5, patience=7)
    logs_mlp  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

    va_f1_mlp = train_model(
        feat_mlp, feat_train_ldr, feat_val_ldr, opt_mlp, criterion_mlp,
        sched_mlp, EarlyStopping(patience=CFG["mlp_patience"], mode="max"),
        CFG["mlp_epochs"], device, logs_mlp, use_mixup=False,
    )
    torch.save(feat_mlp.state_dict(), f"{rd}/best_feature_mlp.pt")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step F — Soft-voting ensemble (weighted by val F1)
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("Step F: Soft-voting ensemble + Stacking + TTA")

    # Get val F1 for all models (re-evaluate on val to be consistent)
    def _va_f1(model, loader, crit):
        _, _, preds, tgts = eval_epoch(model, loader, crit, device)
        return f1_score(tgts, preds, average="macro", zero_division=0)

    va_f1_ct_final   = _va_f1(convtran,    val_loader, criterion)
    va_f1_inc_final  = _va_f1(inception_xl, val_loader, criterion)
    va_f1_ptst_final = _va_f1(patchtst,     val_loader, criterion)
    va_f1_mlp_final  = _va_f1(feat_mlp,    feat_val_ldr, criterion_mlp)

    print(f"\n  Val Macro F1 summary:")
    print(f"    MOMENT         : {va_f1_moment:.4f}")
    print(f"    ConvTran       : {va_f1_ct_final:.4f}")
    print(f"    InceptionXL    : {va_f1_inc_final:.4f}")
    print(f"    PatchTST-Large : {va_f1_ptst_final:.4f}")
    print(f"    XGBoost/Ridge  : {va_f1_xgb:.4f}")
    print(f"    FeatureMLP     : {va_f1_mlp_final:.4f}")

    # Soft weights
    f1s     = np.array([va_f1_moment, va_f1_ct_final, va_f1_inc_final,
                        va_f1_ptst_final, va_f1_xgb, va_f1_mlp_final])
    weights = np.exp(f1s * 10)
    weights /= weights.sum()
    print(f"\n  Ensemble weights: {dict(zip(['MOMENT','ConvTran','InceptXL','PatchTST','XGB','MLP'], np.round(weights,3)))}")

    # TTA probabilities for neural models
    tta = CFG["tta_passes"]
    print(f"\n  Computing TTA ({tta} passes) for neural models...")

    probs_moment = (np.nan_to_num(get_probs_tta(moment_model, test_loader, device, tta), nan=1.0/NUM_CLASSES)
                    if moment_model is not None
                    else np.full((len(y_test), NUM_CLASSES), 1.0/NUM_CLASSES))
    probs_ct     = np.nan_to_num(get_probs_tta(convtran,    test_loader, device, tta), nan=1.0/NUM_CLASSES)
    probs_inc    = np.nan_to_num(get_probs_tta(inception_xl, test_loader, device, tta), nan=1.0/NUM_CLASSES)
    probs_ptst   = np.nan_to_num(get_probs_tta(patchtst,    test_loader, device, tta), nan=1.0/NUM_CLASSES)
    probs_xgb    = np.nan_to_num(xgb_te_probs, nan=1.0/NUM_CLASSES) if xgb_te_probs is not None else np.full((len(y_test), NUM_CLASSES), 1.0/NUM_CLASSES)
    probs_mlp    = np.nan_to_num(get_probs(feat_mlp, feat_test_ldr, device), nan=1.0/NUM_CLASSES)

    probs_ens = (weights[0] * probs_moment + weights[1] * probs_ct +
                 weights[2] * probs_inc    + weights[3] * probs_ptst +
                 weights[4] * probs_xgb    + weights[5] * probs_mlp)
    preds_ens = probs_ens.argmax(axis=1)

    acc_ens, f1_ens, report_ens, cm_ens = compute_metrics(y_test, preds_ens, le)

    # ── Stacking meta-learner ─────────────────────────────────────────────────
    print("\n  Training stacking meta-learner...")

    # Val probabilities for meta-features (not seen during base model training)
    probs_va_moment = (np.nan_to_num(get_probs(moment_model, val_loader, device), nan=1.0/NUM_CLASSES)
                       if moment_model is not None
                       else np.full((len(y_va), NUM_CLASSES), 1.0/NUM_CLASSES))
    probs_va_ct     = np.nan_to_num(get_probs(convtran,    val_loader,  device), nan=1.0/NUM_CLASSES)
    probs_va_inc    = np.nan_to_num(get_probs(inception_xl, val_loader, device), nan=1.0/NUM_CLASSES)
    probs_va_ptst   = np.nan_to_num(get_probs(patchtst,    val_loader,  device), nan=1.0/NUM_CLASSES)
    probs_va_xgb    = np.nan_to_num(xgb_va_probs, nan=1.0/NUM_CLASSES) if xgb_va_probs is not None else np.full((len(y_va), NUM_CLASSES), 1.0/NUM_CLASSES)
    probs_va_mlp    = np.nan_to_num(get_probs(feat_mlp, feat_val_ldr, device), nan=1.0/NUM_CLASSES)

    meta_tr = np.concatenate([probs_va_moment, probs_va_ct, probs_va_inc,
                               probs_va_ptst,  probs_va_xgb, probs_va_mlp], axis=1)
    meta_te = np.concatenate([probs_moment, probs_ct, probs_inc,
                               probs_ptst,  probs_xgb, probs_mlp], axis=1)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler as MetaSc
    meta_sc   = MetaSc().fit(meta_tr)
    meta_lr   = LogisticRegression(
        C=CFG["stacking_C"], class_weight="balanced",
        max_iter=500, solver="lbfgs", multi_class="multinomial",
        random_state=42,
    )
    meta_lr.fit(meta_sc.transform(meta_tr), y_va)
    preds_stack = meta_lr.predict(meta_sc.transform(meta_te))
    acc_stack, f1_stack, report_stack, cm_stack = compute_metrics(y_test, preds_stack, le)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUPER COMPETITOR — FINAL RESULTS")
    print(f"{'='*60}")
    print(f"\n  Model results:")
    print(f"    MOMENT         — Acc: {a_m:.4f}  F1: {f1_m:.4f}")    if moment_model is not None else None
    print(f"    ConvTran       — Acc: {a_ct:.4f}  F1: {f1_ct:.4f}")
    print(f"    InceptionXL    — Acc: {a_inc:.4f}  F1: {f1_inc:.4f}")
    print(f"    PatchTST-Large — Acc: {a_ptst:.4f}  F1: {f1_ptst:.4f}")
    print(f"\n  Ensemble (soft-vote + TTA):")
    print(f"    Acc: {acc_ens:.4f}  Macro F1: {f1_ens:.4f}")
    print(f"\n  Stacking meta-learner:")
    print(f"    Acc: {acc_stack:.4f}  Macro F1: {f1_stack:.4f}")
    print(f"\n{report_stack}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(f"{rd}/competitor_preds.npy",   preds_stack)
    np.save(f"{rd}/competitor_probs.npy",   meta_lr.predict_proba(meta_sc.transform(meta_te)))
    np.save(f"{rd}/competitor_targets.npy", y_test)
    np.save(f"{rd}/competitor_cm.npy",      cm_stack)

    # Also keep soft-vote ensemble
    np.save(f"{rd}/ensemble_softvote_preds.npy",  preds_ens)
    np.save(f"{rd}/ensemble_softvote_probs.npy",  probs_ens)
    np.save(f"{rd}/ensemble_softvote_cm.npy",     cm_ens)

    with open(f"{rd}/results_super_competitor.txt", "w") as fh:
        fh.write("Super Competitor Summary\n" + "="*50 + "\n\n")
        fh.write("Pipeline: MOMENT + ConvTran + InceptionXL + PatchTST-Large + XGBoost + FeatureMLP\n\n")
        fh.write(f"Soft-vote ensemble (TTA={tta}):  Acc={acc_ens:.4f}  F1={f1_ens:.4f}\n")
        fh.write(f"Stacking meta-learner:           Acc={acc_stack:.4f}  F1={f1_stack:.4f}\n\n")
        fh.write(f"Classification Report (Stacking):\n{report_stack}\n")

    print(f"\n  All outputs saved to {rd}/")
    print(f"  Run evaluate.py to generate all figures.\n")
    return acc_stack, f1_stack


if __name__ == "__main__":
    main()
