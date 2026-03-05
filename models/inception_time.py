"""
InceptionTime — best-in-class 1D convolutional model for TSC.
Reference: Fawaz et al. (2020) "InceptionTime: Finding AlexNet for Time Series Classification"

Architecture:
  2 × InceptionBlock (each = 3 InceptionModules + residual shortcut)
  → Global Average Pooling → Dropout → Linear(num_classes)

Input convention (same as course): (B, T, C).
Internally transposed to (B, C, T) for Conv1d.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class InceptionModule(nn.Module):
    """
    One Inception module: three parallel 1D convolutions at different scales
    plus a max-pool branch, all concatenated.

    Input  : (B, in_channels, T)
    Output : (B, nb_filters * 4, T)   — 3 conv branches + 1 maxpool branch
    """

    def __init__(self, in_channels, nb_filters=32,
                 kernel_sizes=(9, 19, 39), use_bottleneck=True):
        super().__init__()

        self.use_bottleneck = use_bottleneck
        bottleneck_ch = nb_filters

        # Bottleneck 1×1 — reduces channels before multi-scale convs
        if use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_ch,
                                        kernel_size=1, bias=False)
        else:
            bottleneck_ch = in_channels

        # Multi-scale parallel convolutions (odd kernel sizes → exact padding)
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_ch, nb_filters,
                      kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])

        # Max-pool branch (on raw input, not bottleneck)
        self.maxpool  = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_mp  = nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)

        n_branches    = len(kernel_sizes) + 1         # 3 conv + 1 maxpool
        out_channels  = nb_filters * n_branches

        self.bn  = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        # Bottleneck
        inp = self.bottleneck(x) if self.use_bottleneck else x

        # Multi-scale branches
        branches = [conv(inp) for conv in self.convs]

        # Max-pool branch
        branches.append(self.conv_mp(self.maxpool(x)))

        out = torch.cat(branches, dim=1)   # (B, nb_filters*4, T)
        return self.act(self.bn(out))


class InceptionBlock(nn.Module):
    """
    3 stacked InceptionModules with a residual (skip) connection.
    Follows the ResNet-style shortcut from the InceptionTime paper.
    """

    def __init__(self, in_channels, nb_filters=32,
                 kernel_sizes=(9, 19, 39), n_modules=3):
        super().__init__()

        out_channels = nb_filters * (len(kernel_sizes) + 1)

        modules = []
        for i in range(n_modules):
            ch_in = in_channels if i == 0 else out_channels
            modules.append(InceptionModule(ch_in, nb_filters, kernel_sizes))
        self.modules_seq = nn.Sequential(*modules)

        # Residual shortcut: 1×1 conv to match channel dimensions
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.modules_seq(x)
        return self.act(out + res)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class InceptionTime(nn.Module):
    """
    Full InceptionTime classifier.

    Input  : x   (B, T, C)   — tslearn/course standard format
             mask (B, T, C)  — binary mask (accepted but not used by conv model)
    Output : logits (B, num_classes)

    Parameters
    ----------
    n_channels   : number of input channels  (6 for LSST)
    num_classes  : number of output classes  (14 for LSST)
    nb_filters   : base filter count per Inception branch  (32 → 128 total)
    kernel_sizes : tuple of kernel sizes for the 3 parallel conv branches
    n_blocks     : number of InceptionBlocks (each has 3 modules + shortcut)
    dropout      : dropout rate before the final linear layer
    """

    def __init__(self, n_channels=6, num_classes=14,
                 nb_filters=32, kernel_sizes=(9, 19, 39),
                 n_blocks=2, dropout=0.2):
        super().__init__()

        out_channels = nb_filters * (len(kernel_sizes) + 1)   # 32*4 = 128

        blocks = []
        for i in range(n_blocks):
            ch_in = n_channels if i == 0 else out_channels
            blocks.append(InceptionBlock(ch_in, nb_filters, kernel_sizes))
        self.blocks = nn.Sequential(*blocks)

        # Global Average Pooling over time → (B, out_channels)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_channels, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x, mask=None):
        """
        Return GAP features (B, out_channels) — used for t-SNE visualization.
        """
        x = x.transpose(1, 2)           # (B, T, C) → (B, C, T)
        x = self.blocks(x)              # (B, out_ch, T)
        return self.gap(x).squeeze(-1)  # (B, out_ch)

    def forward(self, x, mask=None):
        """
        mask is accepted for API consistency but not used by the conv model.
        """
        feats  = self.encode(x, mask)    # (B, out_ch)
        feats  = self.dropout(feats)
        return self.classifier(feats)    # (B, num_classes)
