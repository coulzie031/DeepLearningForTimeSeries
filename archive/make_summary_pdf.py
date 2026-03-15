"""
make_summary_pdf.py -- Generate SUMMARY.pdf for the repo.
4-page technical summary of the LSST classification project.

Usage:  py make_summary_pdf.py
"""

from fpdf import FPDF, XPos, YPos


# ── Colors ────────────────────────────────────────────────────────────────────
PURPLE = (108, 92, 158)
DARK   = (30,  30,  30)
GRAY   = (100, 100, 100)
LIGHT  = (242, 242, 248)
WHITE  = (255, 255, 255)
GREEN  = (55,  168, 104)
BLUE   = (76,  114, 176)


class PDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*PURPLE)
        self.rect(0, 0, 210, 8, "F")
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*WHITE)
        self.set_y(1.5)
        self.cell(0, 5, "LSST Time Series Classification -- Project Summary",
                  align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK)
        self.ln(4)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")

    # ── Typography helpers ────────────────────────────────────────────────────

    def h1(self, text):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*PURPLE)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*PURPLE)
        self.set_line_width(0.4)
        self.line(18, self.get_y(), 192, self.get_y())
        self.ln(3)
        self.set_text_color(*DARK)

    def h2(self, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*PURPLE)
        self.ln(2)
        self.cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK)

    def body(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5, text)

    def bullet(self, text, indent=5):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*DARK)
        x = 18 + indent
        self.set_x(x - 4)
        self.cell(4, 5, "-", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_x(x)
        self.multi_cell(174 - indent, 5, text)

    def code(self, text):
        self.set_fill_color(*LIGHT)
        self.set_font("Courier", "", 8)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 4.8, text, fill=True)
        self.set_text_color(*DARK)
        self.ln(1)

    def kv_row(self, key, val, bold_key=True):
        self.set_font("Helvetica", "B" if bold_key else "", 8.5)
        self.set_text_color(*PURPLE if bold_key else DARK)
        self.cell(30, 5.5, key, new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*DARK)
        self.cell(0, 5.5, val, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def table_header(self, cols):
        """cols: list of (label, width)"""
        self.set_fill_color(*PURPLE)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 8)
        for label, w in cols:
            self.cell(w, 6, label, fill=True,
                      new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.ln(6)

    def table_row_data(self, cells_widths, shade=False):
        """cells_widths: list of (text, width)"""
        self.set_fill_color(*(LIGHT if shade else WHITE))
        self.set_text_color(*DARK)
        self.set_font("Helvetica", "", 8)
        for text, w in cells_widths:
            self.cell(w, 5.5, str(text), fill=True,
                      new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.ln(5.5)


# ─────────────────────────────────────────────────────────────────────────────
# Page builders
# ─────────────────────────────────────────────────────────────────────────────

def page_cover(pdf):
    pdf.add_page()

    # Header banner
    pdf.set_fill_color(*PURPLE)
    pdf.rect(0, 0, 210, 60, "F")

    pdf.set_y(15)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 11, "LSST Time Series Classification",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 13)
    pdf.cell(0, 7, "Project Summary -- Deep Learning for Time Series",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(220, 215, 240)
    pdf.cell(0, 6, "Setting 1 : Adapt a Foundation Model",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_text_color(*DARK)
    pdf.set_y(70)

    # At-a-glance box
    pdf.set_fill_color(*LIGHT)
    pdf.set_draw_color(*PURPLE)
    pdf.set_line_width(0.4)
    pdf.rect(18, pdf.get_y(), 174, 46, "FD")
    pdf.set_xy(22, pdf.get_y() + 4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*PURPLE)
    pdf.cell(0, 6, "Project at a glance", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(22)

    at_a_glance = [
        ("Dataset",    "LSST (PLAsTiCC)  --  14 classes, 6 channels, T=36, ~4000 samples"),
        ("Task",       "Multivariate Time Series Classification"),
        ("Baseline",   "InceptionTime trained from scratch (no pre-training)"),
        ("Competitor", "MOMENT-1-large fine-tuned + MultiROCKET + MANTIS ensemble"),
        ("Metric",     "Accuracy + Macro F1  (14 classes, imbalanced distribution)"),
        ("Deadline",   "Report: March 16, 2026  |  Defense: March 23, 2026"),
    ]
    for k, v in at_a_glance:
        pdf.set_x(22)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*PURPLE)
        pdf.cell(28, 5.5, k, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 5.5, v, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(8)

    # LSST section
    pdf.h1("1.  The LSST Dataset")
    pdf.body(
        "LSST (Large Synoptic Survey Telescope) comes from the PLAsTiCC challenge "
        "(Photometric LSST Astronomical Time-series Classification). Each sample is a "
        "multi-band light curve: photon flux measurements of an astronomical object "
        "over time in 6 photometric filters spanning UV, optical and infrared wavelengths."
    )
    pdf.ln(3)

    pdf.table_header([("Property", 45), ("Value", 129)])
    rows = [
        ("Time steps (T)",  "36 per channel  (chosen so most instances are not truncated)"),
        ("Channels (C)",    "6  --  u, g, r, i, z, Y photometric bands"),
        ("Classes",         "14  --  {6,15,16,42,52,53,62,64,65,67,88,90,92,95}"),
        ("Train / Test",    "~1500 / ~2459 (train smaller than test -- typical for UEA)"),
        ("Class balance",   "Imbalanced  (SNIa very common, Kilonova rare)"),
        ("Missing values",  "Possible NaN -- sparse irregular observations per filter"),
        ("Loading",         "tslearn: UCR_UEA_datasets().load_dataset('LSST')"),
    ]
    for i, (k, v) in enumerate(rows):
        pdf.table_row_data([(k, 45), (v, 129)], shade=(i % 2 == 0))
    pdf.ln(4)

    pdf.h2("14 astronomical object classes")
    classes = [
        "6=SNIa", "15=TDE", "16=EBE", "42=SNIbc", "52=SNIIp", "53=SNIIn",
        "62=SNII", "64=Kilonova", "65=M-dwarf", "67=uLens-Single",
        "88=AGN", "90=SNIax", "92=Non-Ia", "95=SLSN-I",
    ]
    pdf.set_font("Helvetica", "", 8.5)
    for i, cls in enumerate(classes):
        pdf.set_fill_color(*BLUE)
        pdf.set_text_color(*WHITE)
        w = pdf.get_string_width(cls) + 5
        pdf.cell(w, 5, cls, fill=True,
                 new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_text_color(*DARK)
        pdf.cell(3, 5, "", new_x=XPos.RIGHT, new_y=YPos.TOP)
        if (i + 1) % 5 == 0:
            pdf.ln(7)
            pdf.set_x(18)
    pdf.ln(8)


def page_phase1(pdf):
    pdf.add_page()
    pdf.h1("2.  Course Material Overview  (Phase 1 -- Lab Study)")
    pdf.body(
        "8 lab solution notebooks were studied. Their patterns, architectures "
        "and conventions are directly reused in this project."
    )
    pdf.ln(3)

    pdf.table_header([("Notebook", 50), ("Key architectures", 72), ("Reused in project", 50)])
    rows = [
        ("01 basics / SoftDTW",  "SimpleMLP, SoftDTWLoss",                  "Tensor conventions"),
        ("02 RNN / TCN / xLSTM", "GRU, LSTM, TCN (weight_norm), xLSTM",     "Training loop pattern"),
        ("03a Transformers",      "PatchTST, RevIN, SAM optimizer",           "PatchTSTClassifier, RevIN"),
        ("03b Forecasting",       "Direct, AR, teacher forcing, quantile",    "Reference"),
        ("04 TSC TimesNet",       "TimesNet, InceptionModule, GAP",           "InceptionTime baseline"),
        ("04b TSFM",              "MANTIS-8M, MultiROCKET, TiReX",           "MANTIS + MultiROCKET ensemble"),
        ("05 NeuralODE / INR",    "ODE solver, FiLM conditioning, SIREN",    "Reference"),
        ("06 SSM / S4",           "DiagonalSSM, FFT-conv, ZOH discret.",     "Reference"),
    ]
    for i, (nb, arch, use) in enumerate(rows):
        pdf.set_fill_color(*(LIGHT if i % 2 == 0 else WHITE))
        pdf.set_text_color(*DARK)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(50, 5.5, nb, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(72, 5.5, arch, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*PURPLE)
        pdf.cell(50, 5.5, use, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*DARK)
    pdf.ln(5)

    pdf.h2("Key shared conventions  (same across all notebooks)")
    items = [
        "batch_first=True in ALL TransformerEncoderLayer, RNN, GRU, LSTM",
        "Input tensors: (B, T, C)  --  channels last, tslearn standard",
        "Optimizer: AdamW / Adam  |  Loss: CrossEntropyLoss or LabelSmoothing",
        "Training logs: {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}",
        "Normalization: per-instance z-score in __getitem__ (pattern from 04_tsc_sol.ipynb)",
        "Foundation model input: (N, C, T)  --  swap axes from tslearn (N, T, C)",
    ]
    for item in items:
        pdf.bullet(item)
    pdf.ln(4)

    pdf.h2("Foundation model recipe from 04b_tsfm_sol.ipynb")
    pdf.code(
        "from mantis.architecture import Mantis8M\n"
        "from mantis.trainer import MantisTrainer\n"
        "network = Mantis8M(device='cpu').from_pretrained('paris-noah/Mantis-8M')\n"
        "model   = MantisTrainer(network=network, device='cpu')\n"
        "Z_train = model.transform(X_train)   # X: (N, C, T) format\n\n"
        "from aeon.transformations.collection.convolution_based import MultiRocket\n"
        "rocket = MultiRocket(n_kernels=512).fit(X_train)\n"
        "Z_mr   = rocket.transform(X_test)\n\n"
        "# Feature stacking + LogisticRegression linear probe\n"
        "Z = np.concatenate([Z_mantis, Z_rocket], axis=1)\n"
        "LogisticRegression(l1_ratio=1.).fit(Z_train, y_train)"
    )
    pdf.ln(3)

    pdf.h1("3.  Strategy  (Phase 2)")
    pdf.body(
        "Two deliverables: (1) a decent baseline trained from scratch using InceptionTime, "
        "and (2) a strong competitor using MOMENT-1-large fine-tuned with a 3-phase "
        "progressive strategy, further boosted by a feature-stacking ensemble with "
        "MultiROCKET and MANTIS embeddings."
    )
    pdf.ln(2)

    pdf.h2("3.1  Why MOMENT-1-large?")
    pdf.table_header([("Criterion", 52), ("MOMENT", 36), ("MANTIS", 36), ("Chronos/TimesFM", 48)])
    comp_rows = [
        ("Pre-trained on TSC",      "Yes (MOSL)",  "No (SSL)",    "No (forecasting)"),
        ("Multivariate native",     "Yes",          "Yes",         "No"),
        ("Short series (T=36)",     "Yes",          "Medium",      "Problematic"),
        ("Fine-tunable (PyTorch)",  "Yes",          "Partial",     "Difficult"),
        ("Classification head",     "Native",       "Not native",  "Not native"),
    ]
    for i, (crit, m, ma, c) in enumerate(comp_rows):
        shade = (i % 2 == 0)
        pdf.set_fill_color(*(LIGHT if shade else WHITE))
        pdf.set_text_color(*DARK)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(52, 5.5, crit, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        for val, w in [(m, 36), (ma, 36), (c, 48)]:
            good = "Yes" in val or "Native" in val
            bad  = "No" in val or "Not" in val
            pdf.set_text_color(*(GREEN if good else (GRAY if bad else PURPLE)))
            pdf.set_font("Helvetica", "B" if good else "", 8)
            pdf.cell(w, 5.5, val, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_text_color(*DARK)
        pdf.ln(5.5)
    pdf.ln(3)


def page_phase2(pdf):
    pdf.add_page()

    pdf.h2("3.2  NaN Handling")
    pdf.body(
        "LSST observations are sparse (each filter observes at different times). "
        "Three-step fill strategy, then a binary mask is kept as auxiliary information:"
    )
    pdf.code(
        "mask = (~np.isnan(X)).astype(float32)    # 1=observed, 0=NaN\n"
        "# Per channel per sample:\n"
        "#   1. Forward-fill  (last known value propagated forward)\n"
        "#   2. Backward-fill (first known value propagated backward)\n"
        "#   3. Zero-fill     (if entire channel is NaN)\n"
        "# Binary mask passed alongside X to inform models of observation gaps."
    )

    pdf.h2("3.3  Data Augmentation  (training only)")
    aug_rows = [
        ("Jittering",         "Gaussian noise       sigma = 0.05"),
        ("Magnitude scaling", "Uniform scale        s ~ U[0.8, 1.2]"),
        ("Channel dropout",   "Zero one channel     p = 0.2"),
        ("Time warping",      "Resample at warp     w ~ U[0.9, 1.1]"),
        ("MixUp",             "Batch-level          alpha = 0.4,  interpolated loss"),
    ]
    pdf.table_header([("Technique", 50), ("Description", 122)])
    for i, (t, d) in enumerate(aug_rows):
        pdf.table_row_data([(t, 50), (d, 122)], shade=(i % 2 == 0))
    pdf.ln(4)

    pdf.h2("3.4  MOMENT Fine-Tuning Strategy  (3 phases)")
    phase_rows = [
        ("1 -- Linear probing",    "5",   "Encoder frozen, head only",              "lr = 1e-3"),
        ("2 -- Gradual unfreeze",  "30",  "Last 2 Transformer blocks + norm",       "enc 1e-5 / head 1e-3"),
        ("3 -- Full fine-tuning",  "80",  "Entire backbone + head",                 "enc 1e-5 / head 5e-4"),
    ]
    pdf.table_header([("Phase", 55), ("Ep.", 16), ("Description", 78), ("LR", 23)])
    for i, (ph, ep, desc, lr) in enumerate(phase_rows):
        pdf.set_fill_color(*(LIGHT if i % 2 == 0 else WHITE))
        pdf.set_text_color(*DARK)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(55, 5.5, ph, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(16, 5.5, ep, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(78, 5.5, desc, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.cell(23, 5.5, lr, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.h2("3.5  Ensemble  (soft voting, weighted by val accuracy)")
    ens_rows = [
        ("MOMENT-1-large fine-tuned",      "Primary foundation model  (d_emb = 1024)"),
        ("PatchTST (channel-independent)", "Diverse backbone for ensemble diversity"),
        ("FeatureMLP on stacked features", "MOMENT emb + MultiROCKET + MANTIS emb"),
    ]
    for name, role in ens_rows:
        pdf.set_x(22)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*PURPLE)
        pdf.cell(82, 5.5, name, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*GRAY)
        pdf.cell(0, 5.5, "-- " + role, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*DARK)
    pdf.ln(3)

    pdf.h1("4.  Implementation Details")

    pdf.h2("4.1  Baseline -- InceptionTime")
    pdf.body(
        "InceptionTime (Fawaz et al. 2020): parallel multi-scale 1D convolutions, "
        "ResNet-style skip connections, Global Average Pooling -> Linear(14)."
    )
    pdf.code(
        "InceptionTime(\n"
        "    n_channels=6,          # 6 LSST photometric filters\n"
        "    num_classes=14,\n"
        "    nb_filters=32,         # 3 branches x 32 + maxpool branch = 128 ch\n"
        "    kernel_sizes=(9,19,39),# odd kernels -> exact output length\n"
        "    n_blocks=2,            # 2 x (3 InceptionModules + residual)\n"
        "    dropout=0.2,\n"
        ")\n"
        "# Optimizer: AdamW lr=1e-3  |  Scheduler: ReduceLROnPlateau (max acc)\n"
        "# Loss: LabelSmoothingCrossEntropy(smoothing=0.1)\n"
        "# Early stopping patience=20  |  MixUp alpha=0.4  |  grad_clip=1.0"
    )
    pdf.ln(2)

    pdf.h2("4.2  Strong Competitor -- MOMENT-1-large")
    pdf.code(
        "# HuggingFace: AutonLab/MOMENT-1-large\n"
        "# LSST T=36 padded to 40 (next multiple of MOMENT patch_len=8)\n"
        "# Custom head: LayerNorm(1024) -> Linear(1024,256) -> GELU -> Linear(256,14)\n"
        "# Differential LR: encoder x 1e-5, head x 1e-3\n"
        "# CosineAnnealingLR, early stopping patience=15, label smoothing=0.1"
    )

    pdf.h2("4.3  Hyperparameter comparison")
    hp_rows = [
        ("Optimizer",         "AdamW",               "AdamW (differential LR)"),
        ("Learning rate",     "1e-3",                "head 1e-3 / enc 1e-5"),
        ("LR scheduler",      "ReduceLROnPlateau",   "CosineAnnealingLR"),
        ("Batch size",        "64",                  "32"),
        ("Max epochs",        "200",                 "5 + 30 + 80"),
        ("Early stopping",    "patience = 20",       "patience = 15"),
        ("Label smoothing",   "0.1",                 "0.1"),
        ("Weight decay",      "1e-4",                "1e-2  (full fine-tune)"),
        ("Gradient clipping", "max_norm = 1.0",      "max_norm = 1.0"),
    ]
    pdf.table_header([("Parameter", 60), ("Baseline", 57), ("Competitor", 55)])
    for i, (p, bl, co) in enumerate(hp_rows):
        pdf.table_row_data([(p, 60), (bl, 57), (co, 55)], shade=(i % 2 == 0))
    pdf.ln(4)


def page_impl(pdf):
    pdf.add_page()

    pdf.h1("5.  Repository Structure & Run Instructions")

    pdf.code(
        "Time series/\n"
        "  data/\n"
        "    lsst_dataset.py        # load_lsst(), preprocess(), LSSTPatchDataset,\n"
        "                           #   get_dataloaders() with augmentation\n"
        "  models/\n"
        "    inception_time.py      # InceptionTime: InceptionModule, InceptionBlock\n"
        "    moment_classifier.py   # MOMENTClassifier + PatchTSTClassifier (fallback)\n"
        "  utils.py                 # train_epoch, eval_epoch, EarlyStopping,\n"
        "                           #   LabelSmoothingCE, mixup_data, get_embeddings\n"
        "  train_baseline.py        # Run 1st: InceptionTime training loop\n"
        "  train_competitor.py      # Run 2nd: MOMENT 3-phase + MultiROCKET + ensemble\n"
        "  evaluate.py              # Run 3rd: all figures + classification report\n"
        "  make_summary_pdf.py      # This script\n"
        "  results/\n"
        "    fig_loss_baseline.png      fig_loss_moment.png\n"
        "    fig_confusion_baseline.png fig_confusion_competitor.png\n"
        "    fig_confusion_ensemble.png fig_tsne.png  fig_comparison.png\n"
        "    classification_report.txt  results_summary.txt"
    )
    pdf.ln(3)

    pdf.h2("Installation")
    pdf.code(
        "# Core dependencies\n"
        "pip install torch tslearn scikit-learn scipy matplotlib fpdf2\n\n"
        "# Foundation models (competitor)\n"
        "pip install momentfm       # MOMENT-1-large (~600 MB download)\n"
        "pip install aeon           # MultiROCKET\n"
        "pip install mantis-tsfm   # MANTIS-8M"
    )
    pdf.ln(2)

    pdf.h2("Run order")
    pdf.code(
        "# 1. Baseline (InceptionTime from scratch)\n"
        "py train_baseline.py       # ~30 min CPU / ~5 min GPU\n\n"
        "# 2. Strong competitor\n"
        "py train_competitor.py     # ~2-4h (MOMENT download + 3-phase fine-tune)\n\n"
        "# 3. Generate all figures\n"
        "py evaluate.py             # ~5 min  (t-SNE ~1 min)"
    )
    pdf.ln(3)

    pdf.h2("Output figures")
    figs = [
        ("fig_loss_baseline.png",       "InceptionTime train/val loss + accuracy curves"),
        ("fig_loss_moment.png",         "MOMENT competitor training curves"),
        ("fig_confusion_baseline.png",  "Normalized confusion matrix -- baseline"),
        ("fig_confusion_competitor.png","Normalized confusion matrix -- MOMENT"),
        ("fig_confusion_ensemble.png",  "Normalized confusion matrix -- ensemble"),
        ("fig_tsne.png",                "t-SNE of encoder embeddings colored by class"),
        ("fig_comparison.png",          "Bar chart: accuracy + macro F1 of all models"),
    ]
    pdf.table_header([("File", 80), ("Content", 92)])
    for i, (f, d) in enumerate(figs):
        pdf.set_fill_color(*(LIGHT if i % 2 == 0 else WHITE))
        pdf.set_text_color(*DARK)
        pdf.set_font("Courier", "", 8)
        pdf.cell(80, 5.5, f, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(92, 5.5, d, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    pdf.h1("6.  Expected Results")
    pdf.body(
        "Typical accuracy on LSST for state-of-the-art models ranges from 0.65 to 0.80 "
        "depending on the approach. The ensemble is expected to outperform the baseline "
        "by a significant margin thanks to pre-trained representations."
    )
    pdf.ln(3)

    pdf.table_header([("Model", 80), ("Expected Accuracy", 50), ("Expected Macro F1", 42)])
    expected = [
        ("InceptionTime (Baseline)",             "0.65 - 0.72", "0.55 - 0.65"),
        ("PatchTST (from scratch)",              "0.67 - 0.74", "0.57 - 0.67"),
        ("MOMENT fine-tuned",                    "0.74 - 0.82", "0.68 - 0.78"),
        ("Ensemble (Strong Competitor)",         "0.76 - 0.84", "0.70 - 0.80"),
    ]
    for i, (m, a, f) in enumerate(expected):
        pdf.set_fill_color(*(LIGHT if i % 2 == 0 else WHITE))
        pdf.set_text_color(*DARK)
        pdf.set_font("Helvetica", "B" if i == 3 else "", 8.5)
        pdf.set_text_color(*PURPLE if i == 3 else DARK)
        pdf.cell(80, 6, m, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(*GREEN if i >= 2 else DARK)
        pdf.cell(50, 6, a, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(42, 6, f, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*DARK)
    pdf.ln(4)

    pdf.h2("Key insight: why the competitor dominates")
    items = [
        "MOMENT is pre-trained on MOSL benchmark (including classification tasks like LSST)",
        "3-phase fine-tuning avoids catastrophic forgetting of pre-trained representations",
        "MultiROCKET features capture diverse convolutional patterns at 10,000 kernel scales",
        "Ensemble diversity: MOMENT (Transformer) + PatchTST + MultiROCKET (conv kernels)",
        "Label smoothing + MixUp + augmentation regularize the small LSST training set",
    ]
    for item in items:
        pdf.bullet(item)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def build():
    pdf = PDF()
    page_cover(pdf)
    page_phase1(pdf)
    page_phase2(pdf)
    page_impl(pdf)

    out = "SUMMARY.pdf"
    pdf.output(out)
    print(f"Generated: {out}  ({pdf.page_no()} pages)")


if __name__ == "__main__":
    build()
