import torch, torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        base = models.resnet18(pretrained=True)
        # Freeze early layers — good practice for small datasets
        for name, param in base.named_parameters():
            if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
        base.fc = nn.Linear(512, embed_dim)
        self.encoder = base
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.encoder(x))

class TabularEncoder(nn.Module):
    def __init__(self, in_dim=4, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim)
        )
    def forward(self, x):
        return self.net(x)

class BiomassLTNModel(nn.Module):
    def __init__(self, num_targets=5, embed_dim=256):
        super().__init__()
        self.img_enc = ImageEncoder(embed_dim)
        self.tab_enc = TabularEncoder(embed_dim=embed_dim)

        # Cross-modal attention for fusion (better than simple concat)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU()
        )

        # One prediction head per biomass target (multi-task)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim // 2, 64), nn.ReLU(),
                nn.Linear(64, 1)
            ) for _ in range(num_targets)
        ])

    def forward(self, img, tab):
        img_feat = self.img_enc(img).unsqueeze(1)   # (B, 1, D)
        tab_feat = self.tab_enc(tab).unsqueeze(1)   # (B, 1, D)

        # Cross-attention: image attends to tabular context
        fused_attn, _ = self.attn(img_feat, tab_feat, tab_feat)
        fused_attn = fused_attn.squeeze(1)

        # Concat + MLP fusion
        combined = torch.cat([fused_attn, tab_feat.squeeze(1)], dim=-1)
        shared_feat = self.fusion(combined)

        # Per-target predictions
        preds = torch.cat([head(shared_feat) for head in self.heads], dim=-1)
        return preds, shared_feat  # Return shared_feat for LTN rules