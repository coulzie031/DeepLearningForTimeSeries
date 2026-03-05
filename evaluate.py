"""
evaluate.py — Generate ALL figures and evaluation metrics.

Fixes:
- sklearn TSNE: n_iter -> max_iter
- robust loading of baseline v1/v2 artifacts
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  10,
    "lines.linewidth":  1.8,
})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_npy(path, default=None):
    return np.load(path) if os.path.exists(path) else default

def load_logs(path):
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {k: d[k].tolist() for k in d}

def pick_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(logs, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    epochs = range(1, len(logs["train_loss"]) + 1)

    ax = axes[0]
    ax.plot(epochs, logs["train_loss"], label="Train loss")
    ax.plot(epochs, logs["val_loss"],   label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss curves")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    if "train_acc" in logs and logs["train_acc"]:
        ax.plot(epochs, logs["train_acc"], label="Train acc")
    if "val_acc" in logs and logs["val_acc"]:
        ax.plot(epochs, logs["val_acc"], label="Val acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy curves")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(cm, title, save_path, normalize=True):
    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        vmax = 1.0
    else:
        cm_plot = cm
        vmax = None

    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(8, n * 0.7)))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([str(i) for i in range(n)], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([str(i) for i in range(n)], fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tsne(embeddings, labels, title, save_path, n_classes=14):
    print("  Computing t-SNE (may take ~1 min)...")
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    emb = StandardScaler().fit_transform(embeddings)
    if emb.shape[1] > 50:
        emb = PCA(n_components=50, random_state=42).fit_transform(emb)

    # FIX: sklearn recent versions use max_iter, not n_iter
    tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
    max_iter=1000,          # ✅ compatible sklearn récent
    learning_rate="auto",
    init="pca",
)
    Z = tsne.fit_transform(emb)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab20", n_classes)

    for cls in range(n_classes):
        m = (labels == cls)
        if m.sum() == 0:
            continue
        ax.scatter(
            Z[m, 0], Z[m, 1],
            s=15, alpha=0.7, edgecolors="none",
            c=[cmap(cls)], label=str(cls)
        )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, markerscale=2)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison(model_results, save_path):
    model_results = sorted(model_results, key=lambda x: x[1])
    names = [r[0] for r in model_results]
    accs  = [r[1] for r in model_results]
    f1s   = [r[2] for r in model_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(names) * 0.7 + 1)))

    for ax, values, xlabel in zip(axes, [accs, f1s], ["Accuracy", "Macro F1"]):
        bars = ax.barh(names, values, height=0.6, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
                    va="center", ha="left", fontsize=9)
        ax.set_xlim(0, min(1.0, max(values) + 0.08))
        ax.set_xlabel(xlabel)
        ax.set_title(f"Model comparison — {xlabel}", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  EVALUATE — Generating all figures")
    print(f"{'='*60}\n")

    rd = RESULTS_DIR

    # Baseline logs: v1 or v2
    bl_logs_path = pick_first_existing([
        f"{rd}/logs_baseline_v2.npz",
        f"{rd}/logs_baseline.npz",
    ])
    if bl_logs_path:
        logs_bl = load_logs(bl_logs_path)
        plot_training_curves(logs_bl, "Baseline — Training curves", f"{rd}/fig_loss_baseline.png")
    else:
        print("  [WARN] baseline logs not found — skipping baseline curves.")

    # Competitor logs
    mo_logs_path = pick_first_existing([
        f"{rd}/logs_moment.npz",
        f"{rd}/logs_competitor.npz",
    ])
    if mo_logs_path:
        logs_mo = load_logs(mo_logs_path)
        plot_training_curves(logs_mo, "Competitor — Training curves", f"{rd}/fig_loss_moment.png")
    else:
        print("  [WARN] competitor logs not found — skipping competitor curves.")

    # Confusion baseline
    cm_bl = load_npy(f"{rd}/baseline_cm.npy")
    if cm_bl is not None:
        plot_confusion_matrix(cm_bl, "Confusion — Baseline", f"{rd}/fig_confusion_baseline.png", normalize=True)
    else:
        print("  [WARN] baseline_cm.npy not found — skipping baseline confusion.")

    # Confusion competitor / ensemble
    cm_comp = load_npy(f"{rd}/competitor_cm.npy")
    if cm_comp is not None:
        plot_confusion_matrix(cm_comp, "Confusion — Competitor / Ensemble", f"{rd}/fig_confusion_ensemble.png", normalize=True)
    else:
        print("  [WARN] competitor_cm.npy not found — skipping competitor confusion.")

    # t-SNE
    emb = pick_first_existing([
        f"{rd}/moment_embeddings.npy",
        f"{rd}/baseline_embeddings.npy",
    ])
    lbl = pick_first_existing([
        f"{rd}/moment_emb_labels.npy",
        f"{rd}/baseline_emb_labels.npy",
    ])
    if emb and lbl:
        E = np.load(emb)
        L = np.load(lbl)
        plot_tsne(E, L, "t-SNE of embeddings", f"{rd}/fig_tsne.png")
    else:
        print("  [WARN] embeddings not found — skipping t-SNE.")

    # Scores table
    from sklearn.metrics import accuracy_score, f1_score

    model_results = []

    def score(pred_path, tgt_path, name):
        p = load_npy(pred_path)
        t = load_npy(tgt_path)
        if p is None or t is None:
            return
        model_results.append((
            name,
            accuracy_score(t, p),
            f1_score(t, p, average="macro", zero_division=0)
        ))

    # Baseline (v1/v2 same filenames for preds/targets in your scripts)
    score(f"{rd}/baseline_preds.npy", f"{rd}/baseline_targets.npy", "Baseline")

    # Competitor / ensemble
    score(f"{rd}/moment_preds.npy", f"{rd}/competitor_targets.npy", "MOMENT")
    score(f"{rd}/patchtst_preds.npy", f"{rd}/competitor_targets.npy", "PatchTST")
    score(f"{rd}/competitor_preds.npy", f"{rd}/competitor_targets.npy", "Ensemble")

    if len(model_results) >= 2:
        plot_comparison(model_results, f"{rd}/fig_comparison.png")
    else:
        print("  [WARN] not enough results to plot comparison.")

    # Save summary
    with open(f"{rd}/results_summary.txt", "w") as fh:
        fh.write("LSST — Results Summary\n")
        fh.write("=" * 55 + "\n\n")
        fh.write(f"{'Model':<20}  {'Accuracy':>8}  {'MacroF1':>8}\n")
        fh.write("-" * 55 + "\n")
        for name, acc, f1 in sorted(model_results, key=lambda x: x[1], reverse=True):
            fh.write(f"{name:<20}  {acc:>8.4f}  {f1:>8.4f}\n")

    print(f"\nSaved summary to {rd}/results_summary.txt")
    print(f"All figures saved to {rd}/\n")


if __name__ == "__main__":
    main()