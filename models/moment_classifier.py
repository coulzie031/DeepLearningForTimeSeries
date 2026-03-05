"""
MOMENT foundation model adapted for LSST 14-class classification.

Two implementations:
  1. MOMENTClassifier   — wraps momentfm (AutonLab/MOMENT-1-large)
  2. PatchTSTClassifier — from-scratch PatchTST (fallback if momentfm unavailable)

Follows patterns from:
  03a_transformers_sol.ipynb  — PatchTST, RevIN, TransformerEncoderLayer(batch_first=True)
  04b_tsfm_sol.ipynb          — foundation model loading + linear probing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Constants for MOMENT-1-large
# ─────────────────────────────────────────────────────────────────────────────

MOMENT_PATCH_LEN  = 8      # MOMENT-1-large uses 8-length patches
MOMENT_D_MODEL    = 1024   # MOMENT-1-large embedding dimension
LSST_SEQ_LEN      = 36
# Pad LSST (T=36) to next multiple of MOMENT_PATCH_LEN = 40
MOMENT_SEQ_LEN    = ((LSST_SEQ_LEN + MOMENT_PATCH_LEN - 1) // MOMENT_PATCH_LEN) * MOMENT_PATCH_LEN


# ─────────────────────────────────────────────────────────────────────────────
# 1. MOMENT  (primary competitor)
# ─────────────────────────────────────────────────────────────────────────────

class MOMENTClassifier(nn.Module):
    """
    MOMENT-1-large fine-tuned for LSST 14-class classification.

    Usage
    -----
    model = MOMENTClassifier(num_classes=14, n_channels=6)
    model.load_moment()           # downloads + initializes MOMENT backbone
    model.freeze_encoder()        # → linear probing mode
    model.unfreeze_last_n(n=2)    # → partial fine-tuning
    model.unfreeze_all()          # → full fine-tuning

    Forward
    -------
    logits = model(x, mask)       # x: (B,T,C), mask: (B,T,C)
    emb    = model.encode(x, mask)# (B, d_model) — for t-SNE
    """

    def __init__(self, num_classes=14, n_channels=6, dropout=0.2,
                 model_name="AutonLab/MOMENT-1-large"):
        super().__init__()

        self.num_classes  = num_classes
        self.n_channels   = n_channels
        self.seq_len      = MOMENT_SEQ_LEN
        self.model_name   = model_name
        self._loaded      = False

        # Classification head — will use MOMENT_D_MODEL once backbone loaded
        self.head = nn.Sequential(
            nn.LayerNorm(MOMENT_D_MODEL),
            nn.Dropout(dropout),
            nn.Linear(MOMENT_D_MODEL, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Loading ──────────────────────────────────────────────────────────────

    def load_moment(self, device="cpu"):
        """Download and initialize the MOMENT backbone from HuggingFace."""
        import os
        try:
            from momentfm import MOMENTPipeline
        except ImportError:
            raise ImportError(
                "momentfm not installed. Run: pip install momentfm\n"
                "Then retry, or use the PatchTSTClassifier fallback."
            )

        # HuggingFace token — avoids rate-limit warnings on public models
        hf_token = (os.environ.get("HF_TOKEN")
                    or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        if not hf_token:
            print("  [INFO] No HF_TOKEN env var found — downloading without auth"
                  " (may be slower on first download).")

        print(f"  Loading MOMENT backbone from '{self.model_name}' ...")
        load_kwargs = dict(
            model_kwargs={
                "task_name":  "embedding",   # raw embeddings → custom head
                "seq_len":    self.seq_len,
                "n_channels": self.n_channels,
            },
        )
        if hf_token:
            load_kwargs["token"] = hf_token

        self.backbone = MOMENTPipeline.from_pretrained(self.model_name, **load_kwargs)
        self.backbone.init()
        # Force fp32 — MOMENT loads in bfloat16 on some CUDA configs which causes NaN
        self.backbone = self.backbone.float()
        self._loaded = True

        n_total = sum(p.numel() for p in self.backbone.parameters())
        print(f"  MOMENT loaded. seq_len={self.seq_len}, n_channels={self.n_channels},"
              f" total params={n_total:,}")
        return self

    # ── Freeze / unfreeze helpers ─────────────────────────────────────────────

    def freeze_encoder(self):
        """Freeze backbone → only head trains (linear probing)."""
        if not self._loaded:
            raise RuntimeError("Call load_moment() first.")
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("  Backbone frozen — linear probing mode.")

    def _find_encoder_blocks(self):
        """
        Auto-discover transformer encoder blocks inside the backbone.
        Tries known attribute paths for T5, BERT, GPT-2, standard Transformer.
        Falls back to the largest ModuleList found by enumeration.
        """
        # Candidate paths for common Transformer architectures
        # MOMENT-1-large is T5-based → 'model.encoder.block'
        candidate_paths = [
            "model.encoder.block",       # T5  ← MOMENT-1-large
            "model.encoder.layers",      # standard nn.TransformerEncoder
            "model.transformer.h",       # GPT-2
            "model.encoder.layer",       # BERT / RoBERTa
            "encoder.block",
            "encoder.layers",
            "model.blocks",
            "blocks",
        ]
        for path in candidate_paths:
            try:
                obj = self.backbone
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, nn.ModuleList) and len(obj) > 1:
                    print(f"  Found encoder blocks at: backbone.{path} ({len(obj)} blocks)")
                    return obj, path
            except AttributeError:
                continue

        # Fallback: scan all named modules, pick the largest ModuleList
        best, best_name, best_len = None, "", 0
        for name, mod in self.backbone.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > best_len:
                best, best_name, best_len = mod, name, len(mod)

        if best is not None and best_len > 1:
            print(f"  Found largest ModuleList at backbone.{best_name} ({best_len} elements)")
            return best, best_name

        return None, None

    def _disable_gradient_checkpointing(self):
        """
        Disable gradient checkpointing on the backbone to avoid NaN gradients
        when unfreezing T5 layers. Called before any backbone unfreeze.
        """
        try:
            # Standard HuggingFace API
            self.backbone.model.gradient_checkpointing_disable()
            print("  Gradient checkpointing disabled (HF API).")
            return
        except AttributeError:
            pass
        # Manual fallback: walk all modules and disable
        disabled = 0
        for mod in self.backbone.modules():
            if hasattr(mod, "gradient_checkpointing"):
                mod.gradient_checkpointing = False
                disabled += 1
        if disabled:
            print(f"  Gradient checkpointing disabled on {disabled} module(s).")
        else:
            print("  [INFO] No gradient checkpointing found — nothing to disable.")

    def unfreeze_last_n(self, n=2):
        """
        Unfreeze last n transformer blocks + norms + patch embeddings.
        Robustly discovers the block list via _find_encoder_blocks().
        Logs exactly how many parameters become trainable.
        """
        if not self._loaded:
            raise RuntimeError("Call load_moment() first.")
        # Disable gradient checkpointing BEFORE unfreezing to prevent NaN gradients
        self._disable_gradient_checkpointing()

        # Freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad = False

        blocks, path = self._find_encoder_blocks()

        if blocks is not None:
            n_blocks = len(blocks)
            n_unfreeze = min(n, n_blocks)
            for block in list(blocks)[-n_unfreeze:]:
                for p in block.parameters():
                    p.requires_grad = True

            # Always unfreeze norms, patch embedding, positional encoding
            norm_keywords = ("norm", "patch_embed", "pos_embed",
                             "patch_projection", "value_embedding", "head")
            for name, p in self.backbone.named_parameters():
                if any(k in name.lower() for k in norm_keywords):
                    p.requires_grad = True

            n_trainable = sum(p.numel() for p in self.backbone.parameters()
                              if p.requires_grad)
            n_total     = sum(p.numel() for p in self.backbone.parameters())
            print(f"  Unfrozen last {n_unfreeze}/{n_blocks} encoder blocks"
                  f" + norms/embeddings")
            print(f"  Backbone trainable: {n_trainable:,} / {n_total:,}"
                  f" ({100*n_trainable/n_total:.1f}%)")

            # Show trainable module names — proves which encoder blocks are unfrozen
            trainable_names = [name for name, p in self.backbone.named_parameters()
                               if p.requires_grad]
            n_names = len(trainable_names)
            # Always show first 3 + last 3 to confirm block indices (e.g. encoder.block.22)
            if n_names > 6:
                shown = trainable_names[:3] + ["   ..."] + trainable_names[-3:]
            else:
                shown = trainable_names
            print(f"  Trainable param names ({n_names} tensors, sample):")
            for nm in shown:
                print(f"    {nm}")
        else:
            # Last resort: unfreeze by parameter count (last ~30%)
            all_params = list(self.backbone.parameters())
            n_total_params = sum(p.numel() for p in all_params)
            target = int(n_total_params * 0.30)
            accumulated = 0
            for p in reversed(all_params):
                p.requires_grad = True
                accumulated += p.numel()
                if accumulated >= target:
                    break
            n_trainable = sum(p.numel() for p in self.backbone.parameters()
                              if p.requires_grad)
            print(f"  Last-resort fallback: unfroze {n_trainable:,} params"
                  f" ({100*n_trainable/n_total_params:.1f}%)")

    def unfreeze_all(self):
        """Full fine-tuning: unfreeze entire backbone."""
        if not self._loaded:
            raise RuntimeError("Call load_moment() first.")
        self._disable_gradient_checkpointing()
        for p in self.backbone.parameters():
            p.requires_grad = True
        n_total = sum(p.numel() for p in self.backbone.parameters())
        print(f"  Full backbone unfrozen. Total params: {n_total:,}")

    def get_param_groups(self, lr_head=1e-3, lr_encoder=1e-5):
        """
        Differential LR param groups.
        Only includes backbone params that are actually trainable (requires_grad=True).
        Head always trainable at lr_head.
        """
        backbone_trainable = [p for p in self.backbone.parameters()
                              if p.requires_grad]
        if not backbone_trainable:
            # No backbone params are trainable → head-only mode
            return [{"params": list(self.head.parameters()), "lr": lr_head}]
        return [
            {"params": list(self.head.parameters()),  "lr": lr_head},
            {"params": backbone_trainable,             "lr": lr_encoder},
        ]

    # ── Forward utils ─────────────────────────────────────────────────────────

    def _prepare(self, x, mask):
        """
        Prepare MOMENT input.
        x    : (B, T, C) → (B, C, SEQ_LEN)   padded with zeros
        mask : (B, T, C) → (B, SEQ_LEN)       1=observed, 0=pad or NaN
        """
        B, T, C = x.shape
        device  = x.device

        # Channels-first
        x_t = x.transpose(1, 2)      # (B, C, T)

        # Pad to MOMENT_SEQ_LEN
        if T < self.seq_len:
            x_t = F.pad(x_t, (0, self.seq_len - T))   # (B, C, seq_len)

        # Build global mask (B, seq_len) — mean over channels, then pad 0s
        if mask is not None:
            m = mask.float().mean(dim=2)               # (B, T)
        else:
            m = torch.ones(B, T, device=device)
        if T < self.seq_len:
            m = F.pad(m, (0, self.seq_len - T))        # (B, seq_len)

        return x_t, m

    def encode(self, x, mask=None):
        """
        Return MOMENT embeddings (B, d_model) — used for t-SNE.
        """
        x_enc, input_mask = self._prepare(x, mask)
        out = self.backbone(x_enc=x_enc, input_mask=input_mask)
        emb = out.embeddings.float()                    # ensure fp32
        emb = torch.nan_to_num(emb, nan=0.0, posinf=100.0, neginf=-100.0)  # kill any NaN/Inf
        if emb.dim() == 3:
            emb = emb.mean(dim=1)                       # mean over patches
        return emb                                      # (B, d_model)

    def forward(self, x, mask=None):
        """
        x    : (B, T, C)
        mask : (B, T, C) optional
        Returns logits (B, num_classes)
        """
        emb = self.encode(x, mask)          # (B, d_model)
        return self.head(emb)               # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# 2. RevIN  (from 03a_transformers_sol.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

class RevIN(nn.Module):
    """Reversible instance normalization (Ulyanov et al. / Kim et al.)."""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta  = nn.Parameter(torch.zeros(num_features))
        else:
            self.gamma = None
            self.beta  = None

    def _get_stats(self, x):
        # x: (B, T, C)
        self.mean = x.mean(dim=1, keepdim=True)
        self.std  = x.std(dim=1, keepdim=True) + self.eps

    def forward(self, x, mode):
        if mode == "norm":
            self._get_stats(x)
            x = (x - self.mean) / self.std
            if self.gamma is not None:
                x = x * self.gamma + self.beta
        elif mode == "denorm":
            if self.gamma is not None:
                x = (x - self.beta) / (self.gamma + self.eps)
            x = x * self.std + self.mean
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. PatchTST classifier  (fallback — from 03a_transformers_sol.ipynb patterns)
# ─────────────────────────────────────────────────────────────────────────────

class PatchTSTClassifier(nn.Module):
    """
    Channel-independent PatchTST encoder → MLP classification head.

    Input  : (B, T, C)
    Output : logits (B, num_classes)

    Channel-independent trick (from 03a): reshape (B, T, C) → (B*C, T)
    so all channels share the same Transformer weights.

    Serves as fallback if momentfm is not installed.
    Also used as an additional diverse model in the ensemble.
    """

    def __init__(self, seq_len=36, n_channels=6, num_classes=14,
                 patch_len=6, stride=3,
                 d_model=128, n_heads=4, n_layers=4, d_ff=256,
                 dropout=0.1, dropout_head=0.2):
        super().__init__()

        self.seq_len    = seq_len
        self.n_channels = n_channels
        self.patch_len  = patch_len
        self.stride     = stride
        self.d_model    = d_model

        self.num_patches = (seq_len - patch_len) // stride + 1

        # RevIN — per-instance normalization inside model (03a pattern)
        self.revin = RevIN(n_channels)

        # Patch embedding: patch_len → d_model
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_enc    = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.drop_emb   = nn.Dropout(dropout)

        # Transformer encoder (batch_first=True — always in course notebooks)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, batch_first=True, norm_first=True,
            )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Classification head — aggregates all channels
        self.head = nn.Sequential(
            nn.Linear(d_model * n_channels, 256),
            nn.GELU(),
            nn.Dropout(dropout_head),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_patches(self, x):
        """
        x : (B*C, T)  → (B*C, num_patches, patch_len)
        """
        patches = []
        for i in range(0, self.seq_len - self.patch_len + 1, self.stride):
            patches.append(x[:, i: i + self.patch_len])
        return torch.stack(patches, dim=1)   # (B*C, P, patch_len)

    def encode(self, x, mask=None):
        """
        Return per-class feature vector (B, d_model * n_channels) for t-SNE.
        """
        B, T, C = x.shape

        # RevIN normalize
        x = self.revin(x, "norm")

        # Channel-independent: (B, T, C) → (B*C, T)
        x = x.permute(0, 2, 1).reshape(B * C, T)

        # Patching + embedding
        patches = self._make_patches(x)                # (B*C, P, patch_len)
        emb     = self.patch_proj(patches)             # (B*C, P, d_model)
        emb     = self.drop_emb(emb + self.pos_enc)   # + positional encoding

        # Transformer
        enc    = self.transformer(emb)                 # (B*C, P, d_model)
        pooled = self.norm(enc.mean(dim=1))            # (B*C, d_model)

        # Merge channels: (B, C, d_model) → (B, C*d_model)
        pooled = pooled.reshape(B, C, self.d_model)
        return pooled.reshape(B, C * self.d_model)     # (B, C*d_model)

    def forward(self, x, mask=None):
        feats = self.encode(x, mask)
        return self.head(feats)
