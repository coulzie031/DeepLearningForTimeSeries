"""
ConvTran — Convolutional Transformer for Multivariate Time Series Classification.

Based on: "ConvTran: Improving Position Encoding of Transformers for
Multivariate Time Series Classification" (Foumani et al., 2023).

Adapted for LSST: (B, T=36, C=6) → 14 classes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Absolute Position Encoding (tAPE) — time-series aware
# ─────────────────────────────────────────────────────────────────────────────

class tAPE(nn.Module):
    """Time-series Absolute Position Encoding."""

    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(1, seq_len, d_model)
        pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


# ─────────────────────────────────────────────────────────────────────────────
# Relative Position Encoding (eRPE) — convolutional
# ─────────────────────────────────────────────────────────────────────────────

class eRPE(nn.Module):
    """Efficient Relative Position Encoding via depth-wise convolution."""

    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=padding, groups=d_model, bias=False,
        )
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        out = self.conv(x.transpose(1, 2))   # → (B, d_model, T)
        out = self.bn(out).transpose(1, 2)   # → (B, T, d_model)
        return x + out


# ─────────────────────────────────────────────────────────────────────────────
# ConvTran Encoder Layer
# ─────────────────────────────────────────────────────────────────────────────

class ConvTranLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.erpe  = eRPE(d_model, kernel_size=3)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # Self-attention with eRPE bias
        q = self.erpe(x)
        attn_out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# ConvTran
# ─────────────────────────────────────────────────────────────────────────────

class ConvTran(nn.Module):
    """
    ConvTran for LSST multivariate classification.

    Input : x    (B, T, C)  — raw time series (T=36, C=6)
            mask (B, T, C)  — binary mask (0=missing, 1=observed)

    Output: logits (B, num_classes)

    Architecture
    ------------
    1. Channel-mixing conv stem  (C → d_model)
    2. tAPE position encoding
    3. N × ConvTranLayer  (tAPE + eRPE + multi-head attention + FFN)
    4. Global average pooling
    5. Linear head
    """

    def __init__(
        self,
        n_channels:  int   = 6,
        seq_len:     int   = 36,
        num_classes: int   = 14,
        d_model:     int   = 128,
        n_heads:     int   = 8,
        n_layers:    int   = 3,
        d_ff:        int   = 256,
        dropout:     float = 0.1,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        self.tape   = tAPE(d_model, seq_len, dropout=dropout)
        self.layers = nn.ModuleList([
            ConvTranLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm   = nn.LayerNorm(d_model)
        self.head   = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

    def _make_padding_mask(self, mask):
        """
        mask (B, T, C) → key_padding_mask (B, T): True where ALL channels missing.
        MultiheadAttention expects True = ignore.
        """
        if mask is None:
            return None
        obs = mask.float().mean(dim=2)   # (B, T) — fraction observed per step
        return obs < 0.5                  # True = mostly missing → ignore

    def encode(self, x, mask=None):
        """Returns (B, d_model) embedding for t-SNE / stacking."""
        # x: (B, T, C) → (B, C, T) for conv
        h = self.stem(x.transpose(1, 2)).transpose(1, 2)   # (B, T, d_model)
        h = self.tape(h)
        kpm = self._make_padding_mask(mask)
        for layer in self.layers:
            h = layer(h, key_padding_mask=kpm)
        h = self.norm(h)
        return h.mean(dim=1)   # (B, d_model)

    def forward(self, x, mask=None):
        return self.head(self.encode(x, mask))
