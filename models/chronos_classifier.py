"""
Chronos-T5-Small adapted for LSST 14-class classification.

Chronos (Ansari et al., NeurIPS 2024) is Amazon's univariate forecasting
foundation model.  Adaptation strategy:
  - Process each of the 6 LSST passbands independently through the T5 encoder
  - Concatenate 6 channel embeddings: (B, 6 * d_model)
  - Train a classification head on top

Install: pip install git+https://github.com/amazon-science/chronos-forecasting.git
"""

import torch
import torch.nn as nn


class ChronosClassifier(nn.Module):
    """
    Chronos-T5-Small fine-tuned for LSST 14-class classification.

    Usage
    -----
    model = ChronosClassifier(num_classes=14, n_channels=6)
    model.load_chronos(device='cuda')
    model.freeze_encoder()        # Phase 1: linear probing
    model.unfreeze_last_n(n=4)    # Phase 2: partial fine-tune
    logits = model(x, mask)       # x: (B, T, C)
    """

    def __init__(self, num_classes=14, n_channels=6, dropout=0.2,
                 model_name="amazon/chronos-t5-small"):
        super().__init__()
        self.num_classes = num_classes
        self.n_channels  = n_channels
        self.model_name  = model_name
        self.dropout     = dropout
        self._loaded     = False
        self.emb_dim     = None
        self.head        = nn.Identity()   # rebuilt in load_chronos()

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_chronos(self, device="cpu"):
        from chronos import ChronosPipeline
        print(f"  Loading Chronos backbone from '{self.model_name}' ...")
        self.pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
            device_map=device,
            torch_dtype=torch.float32,
        )
        # Ensure float32 (Chronos loads in bfloat16 by default on GPU)
        self.pipeline.model = self.pipeline.model.float()
        self._device = torch.device(device)
        self._loaded = True

        # d_model from T5 config
        d_model = self.pipeline.model.model.config.d_model
        self.emb_dim = self.n_channels * d_model
        print(f"  Chronos d_model={d_model}  emb_dim={self.emb_dim}")

        # Build head
        d = self.dropout
        self.head = nn.Sequential(
            nn.LayerNorm(self.emb_dim),
            nn.Dropout(d),
            nn.Linear(self.emb_dim, 512),
            nn.GELU(),
            nn.Dropout(d),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(d / 2),
            nn.Linear(128, self.num_classes),
        ).to(self._device)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        n_total = sum(p.numel() for p in self.pipeline.model.parameters())
        print(f"  Chronos loaded — backbone params={n_total:,}  head_input={self.emb_dim}")
        return self

    # ── Freeze / unfreeze ─────────────────────────────────────────────────────

    def freeze_encoder(self):
        for p in self.pipeline.model.parameters():
            p.requires_grad = False
        print("  Backbone frozen — linear probing mode.")

    def unfreeze_last_n(self, n=4):
        for p in self.pipeline.model.parameters():
            p.requires_grad = False
        blocks = self.pipeline.model.model.encoder.block
        n_blocks = len(blocks)
        for block in list(blocks)[-min(n, n_blocks):]:
            for p in block.parameters():
                p.requires_grad = True
        # Always unfreeze norms
        for name, p in self.pipeline.model.named_parameters():
            if any(k in name for k in ("layer_norm", "final_layer_norm")):
                p.requires_grad = True
        n_trainable = sum(p.numel() for p in self.pipeline.model.parameters()
                          if p.requires_grad)
        n_total = sum(p.numel() for p in self.pipeline.model.parameters())
        print(f"  Unfrozen last {min(n,n_blocks)}/{n_blocks} encoder blocks + norms")
        print(f"  Backbone trainable: {n_trainable:,} / {n_total:,}"
              f" ({100*n_trainable/n_total:.1f}%)")

    # ── Forward ───────────────────────────────────────────────────────────────

    def encode(self, x, mask=None):
        """
        x: (B, T, C) — process each channel independently through T5 encoder,
        concatenate: (B, n_channels * d_model).
        """
        B, T, C = x.shape
        channel_embs = []

        for c in range(C):
            xc = x[:, :, c].cpu().float()   # (B, T) — tokenizer needs CPU

            # Chronos tokenization
            input_ids, attention_mask, _ = self.pipeline.tokenizer.input_transform(xc)
            input_ids      = input_ids.to(self._device)
            attention_mask = attention_mask.to(self._device)

            # T5 encoder forward
            enc_out = self.pipeline.model.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden = enc_out.last_hidden_state.float()   # (B, L, d_model)

            # Masked mean pooling
            attn_f = attention_mask.float().unsqueeze(-1)   # (B, L, 1)
            emb_c  = (hidden * attn_f).sum(1) / attn_f.sum(1).clamp(min=1)  # (B, d_model)
            channel_embs.append(emb_c)

        return torch.cat(channel_embs, dim=1)   # (B, n_channels * d_model)

    def forward(self, x, mask=None):
        emb = self.encode(x, mask)
        return self.head(emb)

    # ── State dict override — include pipeline weights ────────────────────────

    def state_dict(self, **kwargs):
        sd = super().state_dict(**kwargs)
        # Also save backbone
        for k, v in self.pipeline.model.state_dict().items():
            sd[f"_backbone.{k}"] = v
        return sd

    def load_state_dict(self, state_dict, strict=True):
        backbone_sd = {k[len("_backbone."):]: v
                       for k, v in state_dict.items() if k.startswith("_backbone.")}
        head_sd     = {k: v for k, v in state_dict.items()
                       if not k.startswith("_backbone.")}
        if backbone_sd:
            self.pipeline.model.load_state_dict(backbone_sd, strict=False)
        super().load_state_dict(head_sd, strict=False)
