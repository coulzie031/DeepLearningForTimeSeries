"""
train_competitor.py — Strong competitor: MOMENT fine-tuning + feature ensemble.

Pipeline:
  Step A — PatchTST
  Step B — MOMENT fine-tuning (3 phases) or InceptionTime fallback
  Step C — MultiROCKET features
  Step D — MANTIS embeddings (optional)
  Step E — Feature-stacking MLP
  Step F — Soft-voting ensemble

v3 fixes:
  - seed_everything()
  - robust class weights (length=14 always)
  - robust sampler (minlength=14)
  - FeatureMLP: shuffle=True when not weighted
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

    # Class imbalance strategy: both|sampler_only|weights_only|none
    "imbalance_strategy":   "sampler_only",

    # Augmentation
    "jitter_sigma":         0.03,
    "scale_range":          (0.9, 1.1),
    "channel_drop_p":       0.15,
    "time_warp_p":          0.2,

    # Loss
    "label_smoothing":      0.1,

    # MixUp
    "use_mixup":            True,
    "mixup_alpha":          0.3,

    # Training — linear probing
    "lp_epochs":            5,
    "lp_lr":                1e-3,

    # Training — gradual unfreeze
    "gu_epochs":            30,
    "gu_n_blocks":          2,
    "gu_lr_head":           1e-3,
    "gu_lr_enc":            1e-5,

    # Training — full fine-tune
    "ft_epochs":            80,
    "ft_lr_head":           5e-4,
    "ft_lr_enc":            1e-5,
    "ft_patience":          15,

    # PatchTST
    "ptst_epochs":          150,
    "ptst_lr":              5e-4,
    "ptst_patience":        20,

    # Use MOMENT backbone (gradient checkpointing disabled to prevent NaN)
    "use_moment":           False,  # MOMENT NaN issue — use InceptionTime-Large instead

    # MultiROCKET
    "rocket_n_kernels":     10000,

    # Feature MLP
    "mlp_hidden":           256,
    "mlp_dropout":          0.3,
    "mlp_epochs":           100,
    "mlp_lr":               1e-3,
    "mlp_patience":         15,

    # Paths
    "results_dir":          "results",
}

NUM_CLASSES = 14
N_CHANNELS  = 6
SEQ_LEN     = 36


# ─────────────────────────────────────────────────────────────────────────────
# Feature stacking MLP
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
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def log_param_groups(optimizer, label=""):
    tag = f" [{label}]" if label else ""
    for i, pg in enumerate(optimizer.param_groups):
        n = sum(p.numel() for p in pg["params"] if p.requires_grad)
        print(f"    opt group {i}{tag}: lr={pg['lr']:.2e}, trainable={n:,} params")


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


# ─────────────────────────────────────────────────────────────────────────────
# Step C: MultiROCKET
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Step D: MANTIS
# ─────────────────────────────────────────────────────────────────────────────

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

    T_orig   = X_tr.shape[1]
    T_padded = Xtr.shape[2]
    print(f"  MANTIS shape check:")
    print(f"    pre-padding  (N,T,C): {X_tr.shape}  →  post-padding (N,C,T): {Xtr.shape}")
    print(f"    T: {T_orig} → {T_padded}  (padded with {T_padded-T_orig} edge-repeat steps)")
    assert Xtr.shape[2] % 32 == 0, f"MANTIS requires T divisible by 32, got T={Xtr.shape[2]}"

    print("  MANTIS dry-run (batch=2)...")
    try:
        _ = model.transform(Xtr[:2])
        print("    Dry-run passed.")
    except Exception as dry_err:
        raise RuntimeError(
            f"MANTIS dry-run failed with T={T_padded}: {dry_err}\n"
            f"  Input shape was {Xtr[:2].shape}. Check mantis-tsfm version."
        ) from dry_err

    print("  Extracting MANTIS embeddings...")
    Ztr = model.transform(Xtr)
    Zva = model.transform(Xva)
    Zte = model.transform(Xte)
    print(f"  MANTIS: {Ztr.shape[1]} dims")
    return Ztr, Zva, Zte


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CFG["results_dir"], exist_ok=True)
    seed_everything(CFG["random_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rd     = CFG["results_dir"]

    print_header(f"STRONG COMPETITOR — MOMENT + Ensemble  |  Device: {device}")

    # ── 1) Load & preprocess ──────────────────────────────────────────────────
    X_train_raw, y_train, X_test_raw, y_test, le = load_lsst()

    strategy      = CFG["imbalance_strategy"]
    use_sampler   = strategy in ("both", "sampler_only")
    use_class_w   = strategy in ("both", "weights_only")
    print(f"\nImbalance strategy: '{strategy}'"
          f"  (sampler={use_sampler}, class_weights={use_class_w})")

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

    # Re-derive split arrays (needed for feature extraction)
    from sklearn.model_selection import StratifiedShuffleSplit
    X_train_norm, _ = preprocess(X_train_raw)
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=CFG["val_ratio"], random_state=CFG["random_state"]
    )
    train_idx, val_idx = next(sss.split(X_train_norm, y_train))
    X_tr_norm = X_train_norm[train_idx]
    X_va_norm = X_train_norm[val_idx]
    y_tr      = y_train[train_idx]
    y_va      = y_train[val_idx]

    pipeline_status = {
        "primary_model":      "?",
        "MultiROCKET":        "not run",
        "MANTIS":             "not run",
        "feature_dim":        0,
        "phase1_params":      0,
        "phase2_params":      0,
        "phase3_params":      0,
        "imbalance_strategy": strategy,
        "stacking_type":      "embeddings (encoder .encode() output, not softmax proba)",
    }

    # ── 2) Loss ───────────────────────────────────────────────────────────────
    if use_class_w:
        class_w = compute_class_weights(y_train, num_classes=NUM_CLASSES).to(device)
        print(f"\nClass weights (min={class_w.min():.2f}, max={class_w.max():.2f})")
        criterion = nn.CrossEntropyLoss(
            weight=class_w, label_smoothing=CFG["label_smoothing"],
        )
    else:
        print("\nClass weights: disabled (imbalance_strategy setting)")
        criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])

    # ═══════════════════════════════════════════════════════════════════════════
    # Step A — PatchTST
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("Step A: PatchTST (diverse model for ensemble)")

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

    best_ptst_f1 = train_model(
        patchtst, train_loader, val_loader,
        opt_ptst, criterion, sched_ptst, es_ptst,
        CFG["ptst_epochs"], device, logs_ptst,
        use_mixup=CFG["use_mixup"], mixup_alpha=CFG["mixup_alpha"],
    )
    torch.save(patchtst.state_dict(), f"{rd}/best_patchtst.pt")
    save_logs(logs_ptst, f"{rd}/logs_patchtst.npz")

    _, _, ptst_preds, ptst_targets = eval_epoch(patchtst, test_loader, criterion, device)
    from sklearn.metrics import confusion_matrix as _cm_fn
    np.save(f"{rd}/patchtst_preds.npy", ptst_preds)
    np.save(f"{rd}/patchtst_targets.npy", ptst_targets)
    np.save(f"{rd}/patchtst_cm.npy", _cm_fn(y_test, ptst_preds))
    a_ptst, f1_ptst, _, _ = compute_metrics(y_test, ptst_preds, le)
    print(f"  PatchTST test  —  Acc: {a_ptst:.4f}  Macro F1: {f1_ptst:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step B — MOMENT or fallback InceptionTime
    # ═══════════════════════════════════════════════════════════════════════════

    moment_available = False
    try:
        import momentfm  # noqa
        moment_available = True
    except ImportError:
        print("\n[WARN] momentfm not installed — using InceptionTime as primary model.\n")

    logs_moment = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

    _use_moment = moment_available and CFG.get("use_moment", True)
    primary_model = None  # will be set below

    if _use_moment:
        print_header("Step B: MOMENT fine-tuning (3-phase)")
        try:
            moment = MOMENTClassifier(num_classes=NUM_CLASSES, n_channels=N_CHANNELS, dropout=0.2)
            moment.load_moment(device=str(device))
            moment = moment.to(device)

            # Phase 1 — linear probing (backbone frozen)
            print(f"\n--- Phase 1: Linear probing ({CFG['lp_epochs']} epochs) ---")
            moment.freeze_encoder()
            n_p1 = sum(p.numel() for p in moment.parameters() if p.requires_grad)
            pipeline_status["phase1_params"] = n_p1

            opt_lp = torch.optim.AdamW([p for p in moment.parameters() if p.requires_grad],
                                       lr=CFG["lp_lr"], weight_decay=1e-4)
            log_param_groups(opt_lp, "Phase 1")
            sched_lp = torch.optim.lr_scheduler.CosineAnnealingLR(opt_lp, T_max=CFG["lp_epochs"], eta_min=1e-6)
            train_model(moment, train_loader, val_loader, opt_lp, criterion, sched_lp, None,
                        CFG["lp_epochs"], device, logs_moment, use_mixup=False)

            # Phase 2 — gradual unfreeze (gradient checkpointing disabled inside unfreeze_last_n)
            print(f"\n--- Phase 2: Gradual unfreeze ({CFG['gu_epochs']} epochs) ---")
            moment.unfreeze_last_n(n=CFG["gu_n_blocks"])
            n_p2 = sum(p.numel() for p in moment.parameters() if p.requires_grad)
            pipeline_status["phase2_params"] = n_p2

            param_groups = moment.get_param_groups(lr_head=CFG["gu_lr_head"], lr_encoder=CFG["gu_lr_enc"])
            opt_gu = torch.optim.AdamW(param_groups, weight_decay=1e-4)
            log_param_groups(opt_gu, "Phase 2")
            sched_gu = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gu, T_max=CFG["gu_epochs"], eta_min=1e-7)
            es_gu = EarlyStopping(patience=10, mode="max")
            train_model(moment, train_loader, val_loader, opt_gu, criterion, sched_gu, es_gu,
                        CFG["gu_epochs"], device, logs_moment,
                        use_mixup=CFG["use_mixup"], mixup_alpha=CFG["mixup_alpha"])

            # Phase 3 — full fine-tune
            print(f"\n--- Phase 3: Full fine-tuning ({CFG['ft_epochs']} epochs) ---")
            moment.unfreeze_all()
            n_p3 = sum(p.numel() for p in moment.parameters() if p.requires_grad)
            pipeline_status["phase3_params"] = n_p3

            param_groups = moment.get_param_groups(lr_head=CFG["ft_lr_head"], lr_encoder=CFG["ft_lr_enc"])
            opt_ft = torch.optim.AdamW(param_groups, weight_decay=1e-2)
            log_param_groups(opt_ft, "Phase 3")
            sched_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=CFG["ft_epochs"], eta_min=1e-8)
            es_ft = EarlyStopping(patience=CFG["ft_patience"], mode="max")

            best_moment_f1 = train_model(moment, train_loader, val_loader, opt_ft, criterion, sched_ft, es_ft,
                                         CFG["ft_epochs"], device, logs_moment,
                                         use_mixup=CFG["use_mixup"], mixup_alpha=CFG["mixup_alpha"])

            torch.save(moment.state_dict(), f"{rd}/best_moment.pt")
            save_logs(logs_moment, f"{rd}/logs_moment.npz")

            primary_model = moment
            primary_name  = "MOMENT"
            pipeline_status["primary_model"] = "MOMENT-1-large (fine-tuned, 3-phase)"
            primary_f1 = best_moment_f1

        except Exception as moment_err:
            print(f"\n  [WARN] MOMENT failed: {moment_err}")
            print("  Falling back to InceptionTime-Large.\n")
            primary_model = None  # triggers fallback below

    if primary_model is None:
        reason = ("use_moment=False" if not CFG.get("use_moment", True)
                  else ("MOMENT failed — fallback" if _use_moment else "momentfm missing"))
        print_header(f"Step B: InceptionTime-Large (primary — {reason})")

        primary_model = InceptionTime(
            n_channels=N_CHANNELS, num_classes=NUM_CLASSES,
            nb_filters=64, kernel_sizes=(9, 19, 39), n_blocks=3, dropout=0.2,
        ).to(device)
        print(f"InceptionTime-Large: {sum(p.numel() for p in primary_model.parameters()):,} params\n")

        opt_fb   = torch.optim.AdamW(primary_model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched_fb = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fb, T_max=200, eta_min=1e-6)
        es_fb    = EarlyStopping(patience=25, mode="max")

        primary_f1 = train_model(primary_model, train_loader, val_loader, opt_fb, criterion, sched_fb, es_fb,
                                 200, device, logs_moment,
                                 use_mixup=CFG["use_mixup"], mixup_alpha=CFG["mixup_alpha"])
        torch.save(primary_model.state_dict(), f"{rd}/best_moment.pt")
        save_logs(logs_moment, f"{rd}/logs_moment.npz")

        primary_name = "InceptionTime-Large"
        pipeline_status["primary_model"] = f"InceptionTime-Large ({reason})"

    # Save primary predictions
    _, _, prim_preds, prim_targets = eval_epoch(primary_model, test_loader, criterion, device)
    np.save(f"{rd}/moment_preds.npy", prim_preds)
    np.save(f"{rd}/moment_targets.npy", prim_targets)
    np.save(f"{rd}/moment_cm.npy", _cm_fn(y_test, prim_preds))
    a_prim, f1_prim, _, _ = compute_metrics(y_test, prim_preds, le)
    print(f"  {primary_name} test  —  Acc: {a_prim:.4f}  Macro F1: {f1_prim:.4f}")

    # Save embeddings (t-SNE)
    try:
        emb_te, lbl_te = get_embeddings(primary_model, test_loader, device)
        np.save(f"{rd}/moment_embeddings.npy", emb_te)
        np.save(f"{rd}/moment_emb_labels.npy", lbl_te)
    except Exception as e:
        print(f"  [WARN] Could not extract embeddings: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Steps C/D — Feature extraction
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("Steps C/D: Feature extraction")

    Z_mr_tr, Z_mr_va, Z_mr_te = extract_rocket_features(
        X_tr_norm, X_va_norm, X_test_norm,
        n_kernels=CFG["rocket_n_kernels"],
    )

    Z_ma_tr = Z_ma_va = Z_ma_te = None
    try:
        Z_ma_tr, Z_ma_va, Z_ma_te = extract_mantis_features(X_tr_norm, X_va_norm, X_test_norm)
    except Exception as e:
        print(f"  [WARN] MANTIS extraction failed: {e}")
        print("  Continuing without MANTIS features.")

    print("  Extracting primary model embeddings (NOT class proba)...")
    Z_pm_tr, _ = get_embeddings(primary_model, train_loader, device)
    Z_pm_va, _ = get_embeddings(primary_model, val_loader,   device)
    Z_pm_te, _ = get_embeddings(primary_model, test_loader,  device)
    print(f"  Primary embedding dim: {Z_pm_tr.shape[1]}")

    feat_parts_tr = [Z_pm_tr]
    feat_parts_va = [Z_pm_va]
    feat_parts_te = [Z_pm_te]

    if Z_mr_tr is not None:
        feat_parts_tr.append(Z_mr_tr); feat_parts_va.append(Z_mr_va); feat_parts_te.append(Z_mr_te)
        pipeline_status["MultiROCKET"] = f"OK ({Z_mr_tr.shape[1]} dims)"
        print(f"  + MultiROCKET ({Z_mr_tr.shape[1]} dims)")

    if Z_ma_tr is not None:
        feat_parts_tr.append(Z_ma_tr); feat_parts_va.append(Z_ma_va); feat_parts_te.append(Z_ma_te)
        pipeline_status["MANTIS"] = f"OK ({Z_ma_tr.shape[1]} dims)"
        print(f"  + MANTIS ({Z_ma_tr.shape[1]} dims)")

    feat_tr = np.concatenate(feat_parts_tr, axis=1).astype(np.float32)
    feat_va = np.concatenate(feat_parts_va, axis=1).astype(np.float32)
    feat_te = np.concatenate(feat_parts_te, axis=1).astype(np.float32)
    pipeline_status["feature_dim"] = feat_tr.shape[1]
    print(f"  Total feature dim: {feat_tr.shape[1]}")

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    sk_scaler = StandardScaler().fit(feat_tr)
    feat_tr_s = sk_scaler.transform(feat_tr).astype(np.float32)
    feat_va_s = sk_scaler.transform(feat_va).astype(np.float32)
    feat_te_s = sk_scaler.transform(feat_te).astype(np.float32)

    norm_before = np.linalg.norm(feat_tr, axis=1).mean()
    norm_after  = np.linalg.norm(feat_tr_s, axis=1).mean()
    print(f"  Feature L2 norm: before scaling={norm_before:.1f}  after={norm_after:.1f}")

    # Ridge classifier on MultiROCKET features only (fast, often SOTA on small datasets)
    ridge_clf = None
    va_f1_ridge = 0.0
    ridge_va_probs = None
    ridge_te_probs = None
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
        return DL(ds, batch_size=bs, shuffle=True)  # FIX: was shuffle=False

    feat_train_loader = make_feat_loader(feat_tr_s, y_tr, weighted=True,  bs=64)
    feat_val_loader   = make_feat_loader(feat_va_s, y_va, weighted=False, bs=64)
    feat_test_loader  = make_feat_loader(feat_te_s, y_test, weighted=False, bs=64)

    in_dim   = feat_tr.shape[1]
    feat_mlp = FeatureMLP(in_dim=in_dim, hidden=CFG["mlp_hidden"], num_classes=NUM_CLASSES, dropout=CFG["mlp_dropout"]).to(device)
    print(f"FeatureMLP: {sum(p.numel() for p in feat_mlp.parameters()):,} params\n")

    mlp_class_w = compute_class_weights(y_tr, num_classes=NUM_CLASSES).to(device)
    criterion_mlp = nn.CrossEntropyLoss(weight=mlp_class_w, label_smoothing=CFG["label_smoothing"])

    opt_mlp   = torch.optim.AdamW(feat_mlp.parameters(), lr=CFG["mlp_lr"], weight_decay=1e-4)
    sched_mlp = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_mlp, mode="max", factor=0.5, patience=7, min_lr=1e-6)
    es_mlp    = EarlyStopping(patience=CFG["mlp_patience"], mode="max")
    logs_mlp  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

    best_mlp_f1 = train_model(
        feat_mlp, feat_train_loader, feat_val_loader,
        opt_mlp, criterion_mlp, sched_mlp, es_mlp,
        CFG["mlp_epochs"], device, logs_mlp, use_mixup=False
    )
    torch.save(feat_mlp.state_dict(), f"{rd}/best_feature_mlp.pt")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step F — Soft-voting ensemble
    # ═══════════════════════════════════════════════════════════════════════════
    print_header("Step F: Soft-voting ensemble")

    _, _, prim_va_preds, prim_va_targets = eval_epoch(primary_model, val_loader, criterion, device)
    va_f1_prim = f1_score(prim_va_targets, prim_va_preds, average="macro", zero_division=0)

    _, _, ptst_va_preds, ptst_va_targets = eval_epoch(patchtst, val_loader, criterion, device)
    va_f1_ptst = f1_score(ptst_va_targets, ptst_va_preds, average="macro", zero_division=0)

    _, _, mlp_va_preds, mlp_va_targets = eval_epoch(feat_mlp, feat_val_loader, criterion_mlp, device)
    va_f1_mlp = f1_score(mlp_va_targets, mlp_va_preds, average="macro", zero_division=0)

    print(f"  Val Macro F1 — {primary_name}: {va_f1_prim:.4f}")
    print(f"  Val Macro F1 — PatchTST:       {va_f1_ptst:.4f}")
    print(f"  Val Macro F1 — FeatureMLP:     {va_f1_mlp:.4f}")
    print(f"  Val Macro F1 — Ridge(ROCKET):  {va_f1_ridge:.4f}")

    f1s     = np.array([va_f1_prim, va_f1_ptst, va_f1_mlp, va_f1_ridge])
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

    probs_prim = np.nan_to_num(get_probs(primary_model, test_loader,      device), nan=1.0/NUM_CLASSES)
    probs_ptst = np.nan_to_num(get_probs(patchtst,      test_loader,      device), nan=1.0/NUM_CLASSES)
    probs_mlp  = np.nan_to_num(get_probs(feat_mlp,      feat_test_loader, device), nan=1.0/NUM_CLASSES)
    probs_ridg = np.nan_to_num(ridge_te_probs, nan=1.0/NUM_CLASSES) if ridge_te_probs is not None else np.full_like(probs_prim, 1.0/NUM_CLASSES)

    probs_ens  = (weights[0] * probs_prim + weights[1] * probs_ptst + weights[2] * probs_mlp + weights[3] * probs_ridg)
    preds_ens  = probs_ens.argmax(axis=1)

    acc_ens, f1_ens, report_ens, cm_ens = compute_metrics(y_test, preds_ens, le)

    print(f"\n{'='*60}")
    print(f"  ENSEMBLE TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc_ens:.4f}")
    print(f"  Macro F1 : {f1_ens:.4f}")
    print(f"\n{report_ens}")

    np.save(f"{rd}/competitor_preds.npy",   preds_ens)
    np.save(f"{rd}/competitor_probs.npy",   probs_ens)
    np.save(f"{rd}/competitor_targets.npy", y_test)
    np.save(f"{rd}/competitor_cm.npy",      cm_ens)

    a_prim2, f1_prim2, rep_prim, _ = compute_metrics(y_test, prim_preds, le)
    a_ptst2, f1_ptst2, rep_ptst, _ = compute_metrics(y_test, ptst_preds, le)

    with open(f"{rd}/results_competitor.txt", "w") as f:
        f.write("Strong Competitor Summary\n" + "="*40 + "\n\n")
        f.write("Pipeline status\n" + "-"*40 + "\n")
        f.write(f"  Primary model       : {pipeline_status['primary_model']}\n")
        f.write(f"  MultiROCKET         : {pipeline_status['MultiROCKET']}\n")
        f.write(f"  MANTIS              : {pipeline_status['MANTIS']}\n")
        f.write(f"  Feature dim (total) : {pipeline_status['feature_dim']}\n")
        f.write(f"  Stacking type       : {pipeline_status['stacking_type']}\n")
        f.write(f"  Imbalance strategy  : {pipeline_status['imbalance_strategy']}\n")
        if pipeline_status["phase1_params"] > 0:
            f.write(f"  Phase 1 trainable   : {pipeline_status['phase1_params']:,}\n")
            f.write(f"  Phase 2 trainable   : {pipeline_status['phase2_params']:,}\n")
            f.write(f"  Phase 3 trainable   : {pipeline_status['phase3_params']:,}\n")
        f.write("\nPer-model test scores\n" + "-"*40 + "\n")
        f.write(f"  {primary_name}:\n  Acc={a_prim2:.4f}  Macro F1={f1_prim2:.4f}\n\n")
        f.write(f"  PatchTST:\n  Acc={a_ptst2:.4f}  Macro F1={f1_ptst2:.4f}\n\n")
        f.write(f"  FeatureMLP:\n  Best val Macro F1={best_mlp_f1:.4f}\n\n")
        f.write(f"  Ridge(ROCKET):\n  Val Macro F1={va_f1_ridge:.4f}\n\n")
        f.write("Ensemble:\n")
        f.write(f"  Acc={acc_ens:.4f}  Macro F1={f1_ens:.4f}\n")
        f.write(f"  Weights: {primary_name}={weights[0]:.3f}  PatchTST={weights[1]:.3f}  FeatureMLP={weights[2]:.3f}  Ridge={weights[3]:.3f}\n\n")
        f.write(f"Classification Report (Ensemble):\n{report_ens}\n")

    print(f"\n  All outputs saved to {rd}/")
    print(f"  Run evaluate.py to generate all figures.\n")
    return acc_ens, f1_ens


if __name__ == "__main__":
    main()