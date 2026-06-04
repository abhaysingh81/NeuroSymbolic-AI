import torch
import ltn  
import torch.nn as nn

class HighNDVIPredicate(nn.Module):
    """High NDVI should predict higher total biomass"""
    def forward(self, features, preds, ndvi):
        ndvi_norm = torch.sigmoid(ndvi)           # [0,1]
        total_biomass = preds.sum(dim=-1, keepdim=True)
        expected = ndvi_norm * 2000               # rough domain range
        # Satisfaction: how well preds track NDVI linearly
        sat = torch.exp(-0.001 * (total_biomass - expected).pow(2))
        return sat.squeeze(-1)

class OrderingRule(nn.Module):
    """Soft rule: total biomass >= any individual component"""
    def forward(self, preds):
        total = preds.sum(dim=-1, keepdim=True)
        max_component = preds.max(dim=-1, keepdim=True).values
        # Satisfaction score using fuzzy implication
        margin = total - max_component
        return torch.sigmoid(margin * 0.01).squeeze(-1)

class NonNegativeRule(nn.Module):
    """All biomass predictions must be non-negative"""
    def forward(self, preds):
        # Continuous relaxation of preds >= 0
        return torch.sigmoid(preds).min(dim=-1).values

class HeightBiomassRule(nn.Module):
    """Taller plants should have more total biomass"""
    def forward(self, preds, height):
        total = preds.sum(dim=-1)
        height_norm = torch.sigmoid(height)
        # Pearson-style soft correlation
        t = total - total.mean()
        h = height_norm - height_norm.mean()
        corr = (t * h).sum() / (t.norm() * h.norm() + 1e-8)
        return ((corr + 1) / 2).expand(preds.shape[0])  # [0,1]

class SpeciesConsistencyRule(nn.Module):
    """Within-species predictions should be consistent"""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.species_embed = nn.Embedding(20, embed_dim)
    def forward(self, shared_feat, species_ids):
        sp_feat = self.species_embed(species_ids)
        sim = torch.cosine_similarity(shared_feat, sp_feat)
        return (sim + 1) / 2  # normalize to [0,1]

# -- SATISFACTION AGGREGATOR --
def ltn_sat_loss(rule_satisfactions, weights=None):
    """
    Combines multiple rule satisfactions into a single loss.
    Uses Product T-norm (fuzzy AND): prod(sat_i)
    Lower sat = higher loss.
    """
    if weights is None:
        weights = [1.0] * len(rule_satisfactions)
    sat_stack = torch.stack(rule_satisfactions, dim=-1)  # (B, num_rules)
    # Weighted geometric mean (log-space product)
    log_sat = torch.log(sat_stack.clamp(min=1e-6))
    weighted = (log_sat * torch.tensor(weights, device=log_sat.device)).sum(dim=-1)
    mean_sat = weighted.mean()
    return -mean_sat  # minimize negative satisfaction = maximize satisfaction