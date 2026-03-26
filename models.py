"""
models.py — S-FSCIL model components
  - ViT-B/32 backbone (with partial freezing)
  - ResNet-12 backbone
  - Linear classification head (expandable)
  - Logit fusion MLP
  - Full SFSCILModel assembling all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


# ── Backbone wrappers ─────────────────────────────────────────────────────────

class ViTBackbone(nn.Module):
    """
    CLIP ViT-B/32 used as a trainable feature extractor.
    First `freeze_layers` transformer blocks are frozen during incremental sessions.
    The CLIP image encoder is separate (see CLIPEncoder) and always frozen.
    """

    def __init__(self, freeze_layers=4):
        super().__init__()
        # Load visual encoder from CLIP
        clip_model, _ = clip.load("ViT-B/32", device="cpu")
        self.visual = clip_model.visual
        self.freeze_layers = freeze_layers
        self.embed_dim = 512   # ViT-B/32 output dimension

    def freeze_for_incremental(self):
        """Freeze first freeze_layers transformer blocks."""
        # Always freeze patch embedding and positional embedding
        for name, param in self.visual.named_parameters():
            param.requires_grad = True

        frozen_blocks = [
            self.visual.conv1,
            self.visual.class_embedding,
            self.visual.positional_embedding,
            self.visual.ln_pre,
        ]
        for module in frozen_blocks:
            if isinstance(module, nn.Parameter):
                module.requires_grad_(False)
            else:
                for param in module.parameters():
                    param.requires_grad = False

        for i in range(self.freeze_layers):
            for param in self.visual.transformer.resblocks[i].parameters():
                param.requires_grad = False

    def forward(self, x):
        # Returns [B, embed_dim] feature embedding
        return self.visual(x.type(self.visual.conv1.weight.dtype)).float()


class ResNet12Backbone(nn.Module):
    """
    Standard ResNet-12 used for backbone-controlled comparison experiments.
    embed_dim = 640.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models.resnet import BasicBlock
        self.embed_dim = 640

        def make_layer(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.MaxPool2d(2),
            )

        self.layer1 = make_layer(3, 64)
        self.layer2 = make_layer(64, 160)
        self.layer3 = make_layer(160, 320)
        self.layer4 = make_layer(320, 640)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.dropout(x)


# ── Frozen CLIP encoders ──────────────────────────────────────────────────────

class CLIPEncoder(nn.Module):
    """
    Frozen CLIP image + text encoders used for semantic similarity scoring.
    Completely separate from the trainable ViT backbone.
    """

    def __init__(self, clip_model_name="ViT-B/32", device="cuda"):
        super().__init__()
        self.device = device
        model, self.preprocess = clip.load(clip_model_name, device=device)
        self.image_encoder = model.visual
        self.text_encoder  = model.transformer
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale

        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_image(self, x):
        """Returns L2-normalised image embeddings. x: [B,3,H,W] preprocessed."""
        feat = self.image_encoder(x.to(self.device))
        return F.normalize(feat.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, prompts):
        """
        prompts: list of strings
        Returns L2-normalised text embeddings: [len(prompts), d_clip]
        """
        tokens = clip.tokenize(prompts).to(self.device)
        x = self.token_embedding(tokens).float()
        x = x + self.positional_embedding.float()
        x = x.permute(1, 0, 2)
        x = self.text_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).float()
        x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]
        x = x @ self.text_projection.float()
        return F.normalize(x, dim=-1)


# ── Expandable classification head ───────────────────────────────────────────

class IncrementalClassifier(nn.Module):
    """
    Linear classification head that grows by N units at each session.
    Stores separate weight blocks per session to support zero-padding for SAUD.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.weights = nn.ParameterList()  # one block per session
        self.num_classes = 0

    def add_classes(self, n_new):
        """Append a new weight block for n_new classes."""
        w = nn.Parameter(torch.empty(n_new, self.embed_dim))
        nn.init.kaiming_normal_(w, mode="fan_out")
        self.weights.append(w)
        self.num_classes += n_new

    def forward(self, x):
        """Returns logits [B, total_classes]."""
        W = torch.cat(list(self.weights), dim=0)  # [total_classes, d]
        return x @ W.t()

    def get_session_logits(self, x, session_idx):
        """
        Return logits for a specific session's weight block only.
        Used for zero-padding SAUD components.
        """
        return x @ self.weights[session_idx].t()


# ── Logit fusion MLP ──────────────────────────────────────────────────────────

class LogitFusionMLP(nn.Module):
    """
    2-layer MLP operating on concatenated ell_2-normalised logits from
    base head and previous session head.
    Input:  [B, 2 * num_classes]   (after zero-padding and normalisation)
    Output: [B, num_classes]        (refined distillation logits)

    Note: this MLP refines classification logits and does NOT interact
    with CLIP's embedding space.
    """

    def __init__(self, in_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim // 2),  # out_dim = num_classes
        )

    def forward(self, x):
        return self.net(x)


# ── Full S-FSCIL model ────────────────────────────────────────────────────────

class SFSCILModel(nn.Module):
    """
    Full S-FSCIL model assembling:
      - Trainable backbone (ViT-B/32 or ResNet-12)
      - Expandable classification head
      - Logit fusion MLP for SAUD
      - Frozen CLIP encoders (separate instance, used only for scoring)
    """

    def __init__(self, args, device="cuda"):
        super().__init__()
        self.args = args
        self.device = device

        # Trainable feature extractor
        if args.backbone == "ViT-B/32":
            self.backbone = ViTBackbone(freeze_layers=args.freeze_layers)
            embed_dim = self.backbone.embed_dim
        else:
            self.backbone = ResNet12Backbone()
            embed_dim = self.backbone.embed_dim
        self.embed_dim = embed_dim

        # Expandable classifier
        self.classifier = IncrementalClassifier(embed_dim)

        # Logit fusion MLP — initialised lazily after first class expansion
        self.mlp = None
        self.mlp_hidden = args.mlp_hidden

        # Frozen CLIP (separate from backbone)
        self.clip = CLIPEncoder(args.clip_model, device=device)

        # Session bookkeeping
        self.session = 0
        self.class_offsets = []   # cumulative class counts per session
        self.text_embeddings = {} # cache: session_id -> [C^(t), d_clip]

    # ── Initialisation helpers ────────────────────────────────────────────────

    def add_session_classes(self, n_new):
        """Expand classifier and MLP for a new session."""
        prev_total = self.classifier.num_classes
        self.classifier.add_classes(n_new)
        new_total = self.classifier.num_classes
        self.class_offsets.append(prev_total)

        # Rebuild MLP with updated dimensions
        self.mlp = LogitFusionMLP(
            in_dim=2 * new_total,
            hidden_dim=self.mlp_hidden,
        ).to(self.device)
        self.session += 1

    def cache_text_embeddings(self, class_names, session_id,
                              prompt_templates, ensemble=True):
        """
        Compute and cache CLIP text embeddings for all known classes.
        Uses prompt ensembling if ensemble=True and len(templates) > 1.
        """
        embeddings = []
        for name in class_names:
            if ensemble and len(prompt_templates) > 1:
                # Average over K templates then L2-normalise
                vecs = self.clip.encode_text(
                    [t.format(name) for t in prompt_templates]
                )
                emb = F.normalize(vecs.mean(0, keepdim=True), dim=-1)
            else:
                emb = self.clip.encode_text(
                    [prompt_templates[0].format(name)]
                )
            embeddings.append(emb)
        self.text_embeddings[session_id] = torch.cat(embeddings, dim=0)

    # ── Forward passes ────────────────────────────────────────────────────────

    def encode(self, x):
        """Feature extraction: φ(x) = f_θ(x). [B, d]"""
        return self.backbone(x)

    def classify(self, phi):
        """Full logit vector over all seen classes. [B, C^(t)]"""
        return self.classifier(phi)

    def clip_similarity(self, x_orig, session_id):
        """
        CLIP semantic similarity z_clip: [B, C^(t)].
        Uses the frozen CLIP image encoder on the ORIGINAL (non-augmented) input.
        This is a SEPARATE forward pass from self.encode().
        """
        img_emb = self.clip.encode_image(x_orig)   # [B, d_clip]
        txt_emb = self.text_embeddings[session_id]  # [C^(t), d_clip]
        return img_emb @ txt_emb.t()                # [B, C^(t)]

    # ── SAUD distillation target ──────────────────────────────────────────────

    def build_distillation_target(self, phi, z_clip,
                                  base_model, prev_model,
                                  total_classes):
        """
        Construct ẑ = Norm(ẑ_b) + Norm(ẑ_{t-1}) + Norm(z_clip)
        with zero-padding to total_classes.

        Args:
            phi:           [B, d]  features from current model
            z_clip:        [B, C^(t)]  CLIP similarity vector
            base_model:    frozen SFSCILModel from session 0
            prev_model:    frozen SFSCILModel from session t-1
            total_classes: int = |C^(t)|

        Returns:
            z_hat: [B, total_classes]
        """
        B = phi.size(0)
        device = phi.device

        # Base head logits — zero-pad new class slots
        n_base = base_model.classifier.num_classes
        z_b_raw = base_model.classifier(phi)          # [B, |C_0|]
        z_b = torch.zeros(B, total_classes, device=device)
        z_b[:, :n_base] = z_b_raw

        # Previous session logits — zero-pad new class slots
        n_prev = prev_model.classifier.num_classes
        z_prev_raw = prev_model.classifier(phi)        # [B, |C^(t-1)|]
        z_prev = torch.zeros(B, total_classes, device=device)
        z_prev[:, :n_prev] = z_prev_raw

        # ell_2 normalize each component (zero entries stay zero)
        z_b_n    = F.normalize(z_b,     dim=-1)
        z_prev_n = F.normalize(z_prev,  dim=-1)
        z_clip_n = F.normalize(z_clip,  dim=-1)

        # Fused supervisory target (Eq. z)
        z_hat = z_b_n + z_prev_n + z_clip_n          # [B, total_classes]
        return z_hat

    def freeze_for_incremental(self):
        """Freeze first freeze_layers of backbone for incremental training."""
        if isinstance(self.backbone, ViTBackbone):
            self.backbone.freeze_for_incremental()
