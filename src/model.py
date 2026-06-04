import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    from torchvision import models


# ─────────────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────

class ResidualMLP(nn.Module):
    """MLP block with residual skip-connection."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.net(x) + self.proj(x))


class ImageEncoder(nn.Module):
    """
    EfficientNet-B2 backbone with partial fine-tuning.
    Falls back to ResNet-18 if timm is not installed.
    """

    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        if TIMM_AVAILABLE:
            self.backbone = timm.create_model(
                "efficientnet_b2", pretrained=pretrained, num_classes=0
            )
            feat_dim = self.backbone.num_features  # 1408 for B2
            # Freeze stem + first 4 blocks, fine-tune last 3
            total_blocks = len(list(self.backbone.blocks))
            for i, block in enumerate(self.backbone.blocks):
                for p in block.parameters():
                    p.requires_grad = i >= (total_blocks - 3)
        else:
            self.backbone = models.resnet18(pretrained=pretrained)
            feat_dim = 512
            self.backbone.fc = nn.Identity()
            for name, p in self.backbone.named_parameters():
                if "layer3" not in name and "layer4" not in name:
                    p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # (B, feat_dim) — global pooled by timm
        if feats.dim() > 2:
            feats = feats.mean(dim=[-2, -1])
        return self.proj(feats)


class TabularEncoder(nn.Module):
    """
    Deep MLP for the 5 tabular features:
    [ndvi_scaled, height_scaled, species_enc, month_sin, month_cos]
    """

    def __init__(self, in_dim: int = 5, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            ResidualMLP(in_dim, 64,  128),
            ResidualMLP(128,    256, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalAttention(nn.Module):
    """
    Multi-head cross-attention.
    Image embedding queries the tabular context to focus on
    agronomically relevant feature combinations.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_feat: torch.Tensor, tab_feat: torch.Tensor) -> torch.Tensor:
        # img attends to tab  →  (B, 1, D)
        q = img_feat.unsqueeze(1)
        k = v = tab_feat.unsqueeze(1)
        out, _ = self.attn(q, k, v)
        return self.norm(out.squeeze(1) + img_feat)   # residual


class PredictionHead(nn.Module):
    """Single biomass-target head with uncertainty estimation."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 1)


# ─────────────────────────────────────────────────────────────────────
# 1. NEURAL BASELINE  (Approach 1)
# ─────────────────────────────────────────────────────────────────────

class NeuralBaseline(nn.Module):
    """
    Image + tabular concatenation with MSE loss only.
    No LTN rules, no soft constraints.
    Serves as the neural-only comparison baseline.
    """

    def __init__(self, num_targets: int = 5, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.img_enc = ImageEncoder(embed_dim, pretrained)
        self.tab_enc = TabularEncoder(embed_dim=embed_dim)
        self.fusion  = ResidualMLP(embed_dim * 2, embed_dim, embed_dim // 2)
        self.heads   = nn.ModuleList([PredictionHead(embed_dim // 2) for _ in range(num_targets)])

    def forward(self, img: torch.Tensor, tab: torch.Tensor):
        img_feat = self.img_enc(img)
        tab_feat = self.tab_enc(tab)
        fused    = self.fusion(torch.cat([img_feat, tab_feat], dim=-1))
        preds    = torch.cat([h(fused) for h in self.heads], dim=-1)  # (B, num_targets)
        return preds, fused   # return fused so eval code is uniform


# ─────────────────────────────────────────────────────────────────────
# 2. SYMBOLIC BASELINE  (Approach 2)
# ─────────────────────────────────────────────────────────────────────

class SymbolicBaseline(nn.Module):
    """
    Tabular-only regression using hand-crafted feature interactions.
    Encodes domain rules as explicit polynomial features, then fits
    a small linear model.  No images used.

    Rules hard-coded here:
      - NDVI * height   (interaction)
      - NDVI^2          (nonlinear NDVI effect)
      - species one-hot (fixed species intercept)
      - seasonal (month_sin / month_cos)
    """

    def __init__(self, num_targets: int = 5, num_species: int = 20):
        super().__init__()
        self.num_species = num_species
        # Input dim: 5 raw + ndvi*height, ndvi^2, height^2, species_onehot
        self.in_dim = 5 + 3 + num_species
        self.linear = nn.Linear(self.in_dim, num_targets)
        nn.init.xavier_uniform_(self.linear.weight)

    def _build_features(self, tab: torch.Tensor) -> torch.Tensor:
        """
        tab columns: [ndvi_scaled, height_scaled, species_enc, month_sin, month_cos]
        """
        ndvi    = tab[:, 0:1]
        height  = tab[:, 1:2]
        sp_idx  = tab[:, 2].long().clamp(0, self.num_species - 1)
        month_s = tab[:, 3:4]
        month_c = tab[:, 4:5]

        sp_onehot = F.one_hot(sp_idx, self.num_species).float()
        cross     = ndvi * height                         # interaction
        ndvi_sq   = ndvi ** 2
        height_sq = height ** 2

        return torch.cat([ndvi, height, month_s, month_c,
                          tab[:, 2:3],           # species as float
                          cross, ndvi_sq, height_sq,
                          sp_onehot], dim=-1)

    def forward(self, img: torch.Tensor, tab: torch.Tensor):
        feats = self._build_features(tab)
        preds = self.linear(feats)    # (B, num_targets)
        return preds, feats


# ─────────────────────────────────────────────────────────────────────
# 3. LTN NEURO-SYMBOLIC MODEL  (Approach 3 — primary)
# ─────────────────────────────────────────────────────────────────────

class LTNModel(nn.Module):
    """
    Full multi-modal model with:
      - EfficientNet-B2 image branch
      - ResidualMLP tabular branch
      - Cross-modal attention fusion
      - Shared embedding (returned for LTN rule evaluation)
      - 5 independent prediction heads
    """

    def __init__(self, num_targets: int = 5, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.img_enc  = ImageEncoder(embed_dim, pretrained)
        self.tab_enc  = TabularEncoder(embed_dim=embed_dim)
        self.cross_attn = CrossModalAttention(embed_dim, num_heads=8)

        # After attention: fuse attended image feat + tabular feat
        self.fusion = nn.Sequential(
            ResidualMLP(embed_dim * 2, embed_dim, embed_dim),
            nn.Dropout(0.3),
            ResidualMLP(embed_dim, embed_dim, embed_dim // 2),
        )

        self.heads = nn.ModuleList([PredictionHead(embed_dim // 2) for _ in range(num_targets)])

    def forward(self, img: torch.Tensor, tab: torch.Tensor):
        img_feat  = self.img_enc(img)                       # (B, D)
        tab_feat  = self.tab_enc(tab)                       # (B, D)
        attended  = self.cross_attn(img_feat, tab_feat)     # (B, D)

        shared = self.fusion(torch.cat([attended, tab_feat], dim=-1))  # (B, D/2)
        preds  = torch.cat([h(shared) for h in self.heads], dim=-1)   # (B, T)

        return preds, shared    # shared embedding passed to LTN rule layer


# ─────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────

def build_model(model_type: str, num_targets: int = 5, **kwargs) -> nn.Module:
    """
    model_type: 'ltn' | 'neural' | 'symbolic'
    """
    registry = {
        "ltn":      LTNModel,
        "neural":   NeuralBaseline,
        "symbolic": SymbolicBaseline,
    }
    if model_type not in registry:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(registry)}")
    return registry[model_type](num_targets=num_targets, **kwargs)