from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
# FUZZY OPERATOR LIBRARY
# ─────────────────────────────────────────────────────────────────────

def product_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


def lukasiewicz_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.clamp(a + b - 1, min=0.0)


def godel_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.min(a, b)


def residuum(a: torch.Tensor, b: torch.Tensor, tnorm: str = "product") -> torch.Tensor:
    """
    Fuzzy implication: a → b
    Product residuum: min(1, b/a) with smoothing.
    """
    if tnorm == "product":
        return torch.clamp(b / (a + 1e-8), max=1.0)
    elif tnorm == "lukasiewicz":
        return torch.clamp(1 - a + b, max=1.0, min=0.0)
    else:   # Gödel
        return torch.where(a <= b, torch.ones_like(a), b)


# ─────────────────────────────────────────────────────────────────────
# SATISFACTION AGGREGATOR
# ─────────────────────────────────────────────────────────────────────

class SatAgg(nn.Module):
    """
    Aggregates per-sample, per-rule satisfaction scores into a single
    scalar loss using the pMean generalised aggregator (LTN paper, eq.5).

    p=1  → arithmetic mean   (lenient: one bad rule compensated by others)
    p=2  → quadratic mean    (moderate)
    p→∞  → min               (strict: worst rule dominates)

    Default p=2 balances gradient flow and strictness.
    """

    def __init__(self, p: float = 2.0):
        super().__init__()
        self.p = p

    def forward(
        self,
        satisfactions: List[torch.Tensor],   # list of (B,) tensors in [0,1]
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Returns scalar loss: 1 - mean_satisfaction.
        Minimising this loss maximises rule satisfaction.
        """
        if weights is None:
            weights = [1.0] * len(satisfactions)

        w = torch.tensor(weights, dtype=torch.float32, device=satisfactions[0].device)
        w = w / w.sum()

        # Stack: (B, num_rules)
        sat_stack = torch.stack(satisfactions, dim=1).clamp(1e-6, 1.0)

        # Weighted pMean across rules per sample
        sat_pmean = (w * sat_stack.pow(self.p)).sum(dim=1).pow(1.0 / self.p)  # (B,)

        # Mean over batch → scalar; invert to get loss
        mean_sat = sat_pmean.mean()
        return 1.0 - mean_sat


# ─────────────────────────────────────────────────────────────────────
# INDIVIDUAL RULES
# ─────────────────────────────────────────────────────────────────────

class NDVIBiomassRule(nn.Module):
    """
    R1: High NDVI → high total biomass.

    Grounded as:  sat = σ(β * (total_pred/scale - ndvi))
    where β controls slope and scale normalises biomass to [0,1]-ish range.

    EDA finding: r(NDVI, total_biomass) > 0 for all 5 targets.
    """

    def __init__(self, beta: float = 5.0, scale: float = 1000.0):
        super().__init__()
        self.beta  = beta
        self.scale = scale

    def forward(self, preds: torch.Tensor, ndvi: torch.Tensor) -> torch.Tensor:
        """
        preds : (B, T) — log1p predictions
        ndvi  : (B,)   — scaled NDVI (from tabular col 0)
        """
        total     = preds.exp().sum(dim=-1) - preds.shape[-1]   # undo log1p approx
        total_n   = torch.sigmoid(total / self.scale)            # [0,1]
        ndvi_n    = torch.sigmoid(ndvi)                          # [0,1]
        # Implication: ndvi_n → total_n
        sat = residuum(ndvi_n, total_n, tnorm="product")
        return sat.clamp(0.0, 1.0)


class HeightBiomassRule(nn.Module):
    """
    R2: Taller plants → more total biomass (positive correlation).

    Sat = softplus correlation coefficient normalised to [0,1].
    EDA finding: r(height, total_biomass) ≈ 0.3–0.6 depending on species.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temp = temperature

    def forward(self, preds: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
        total  = preds.sum(dim=-1)              # (B,)
        # Rank correlation proxy (differentiable)
        t_norm = (total  - total.mean())  / (total.std()  + 1e-8)
        h_norm = (height - height.mean()) / (height.std() + 1e-8)
        corr   = (t_norm * h_norm).mean()       # scalar ∈ [-1, 1]
        sat    = (corr + 1) / 2                 # map to [0, 1]
        return sat.expand(preds.shape[0])       # broadcast to (B,)


class OrderingRule(nn.Module):
    """
    R3: Total biomass ≥ each individual component.

    sat_i = σ(margin * (total - component_i))
    Final sat = product over i (fuzzy AND across components).

    EDA finding: Violation rate < 5% empirically → strong prior.
    """

    def __init__(self, margin: float = 0.01):
        super().__init__()
        self.margin = margin

    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        total = preds.sum(dim=-1, keepdim=True)          # (B,1)
        diff  = total - preds                             # (B,T)  should all be ≥ 0
        sat_per_component = torch.sigmoid(self.margin * diff)  # (B,T) ∈ [0,1]
        # Fuzzy AND: product over targets
        sat = sat_per_component.prod(dim=-1)             # (B,)
        return sat


class NonNegativeRule(nn.Module):
    """
    R4: All predictions ≥ 0 (biomass is always non-negative).

    Hardest physical constraint. Penalise negative predictions heavily.
    sat = prod_i σ(γ * pred_i)
    """

    def __init__(self, gamma: float = 10.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        sat_per = torch.sigmoid(self.gamma * preds)   # (B,T)
        return sat_per.prod(dim=-1)                   # (B,)


class SpeciesConsistencyRule(nn.Module):
    """
    R5: Samples from the same species should produce similar shared embeddings.

    Within a batch, compute pairwise cosine similarity between same-species
    pairs and use the mean as satisfaction.

    EDA finding: Inter-species variance > intra-species variance for all targets.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temp = temperature

    def forward(self, shared_feat: torch.Tensor, species_ids: torch.Tensor) -> torch.Tensor:
        B = shared_feat.shape[0]
        # Normalise embeddings
        norm_feat = F.normalize(shared_feat, dim=-1)       # (B, D)
        sim_matrix = torch.mm(norm_feat, norm_feat.t())    # (B, B)

        # Same-species mask
        sp = species_ids.unsqueeze(1)                      # (B,1)
        same_mask = (sp == sp.t()).float()                 # (B,B)
        # Exclude diagonal
        eye = torch.eye(B, device=shared_feat.device)
        same_mask = same_mask * (1 - eye)

        n_pairs = same_mask.sum().clamp(min=1.0)
        mean_sim = (sim_matrix * same_mask).sum() / n_pairs   # scalar [-1,1]
        sat = (mean_sim + 1) / 2                              # [0,1]
        return sat.expand(B)


class SeasonalityRule(nn.Module):
    """
    R6: High NDVI months should correspond to higher predicted biomass.

    Uses the month cyclical encoding (month_sin, month_cos) to derive a
    'growing season score' and checks it is positively correlated with
    total predicted biomass.

    EDA finding: NDVI peaks in spring/summer → biomass peaks follow.
    """

    def forward(self, preds: torch.Tensor, month_sin: torch.Tensor, month_cos: torch.Tensor) -> torch.Tensor:
        # Growing season proxy: month 9-11 (spring) → high score in Southern hemisphere
        season_score = month_sin   # peaks at month 3 (March) and 9 (September)
        total = preds.sum(dim=-1)
        t_norm = (total - total.mean()) / (total.std() + 1e-8)
        s_norm = (season_score - season_score.mean()) / (season_score.std() + 1e-8)
        corr   = (t_norm * s_norm).mean()
        sat    = (corr + 1) / 2
        return sat.expand(preds.shape[0])


# ─────────────────────────────────────────────────────────────────────
# RULE BANK  (all rules + aggregator in one object)
# ─────────────────────────────────────────────────────────────────────

class RuleBank(nn.Module):
    """
    Wraps all 6 rules and the SatAgg.
    Returns (sat_loss, rule_dict) for logging.

    Usage
    -----
    rules = RuleBank(p=2.0)
    sat_loss, info = rules(preds, shared_feat, tab, species_ids)
    total_loss = mse_loss + ltn_lambda * sat_loss
    """

    def __init__(self, p: float = 2.0):
        super().__init__()
        self.r1 = NDVIBiomassRule()
        self.r2 = HeightBiomassRule()
        self.r3 = OrderingRule()
        self.r4 = NonNegativeRule()
        self.r5 = SpeciesConsistencyRule()
        self.r6 = SeasonalityRule()
        self.agg = SatAgg(p=p)

    # Rule weights (higher = more important for training)
    WEIGHTS = {
        "r4_nonneg":     3.0,   # physics constraint — hardest
        "r3_ordering":   2.0,   # agronomy ordering
        "r1_ndvi":       1.5,   # NDVI signal strong (from EDA)
        "r2_height":     1.0,   # height signal moderate
        "r5_species":    0.8,   # consistency soft prior
        "r6_seasonal":   0.5,   # weakest — month encoding noisy
    }

    def forward(
        self,
        preds:       torch.Tensor,    # (B, T)
        shared_feat: torch.Tensor,    # (B, D)
        tab:         torch.Tensor,    # (B, 5)
        species_ids: torch.Tensor,    # (B,)  long
    ):
        ndvi      = tab[:, 0]
        height    = tab[:, 1]
        month_sin = tab[:, 3]
        month_cos = tab[:, 4]

        s1 = self.r1(preds, ndvi)
        s2 = self.r2(preds, height)
        s3 = self.r3(preds)
        s4 = self.r4(preds)
        s5 = self.r5(shared_feat, species_ids)
        s6 = self.r6(preds, month_sin, month_cos)

        rule_dict = {
            "r1_ndvi":    s1.mean().item(),
            "r2_height":  s2.mean().item(),
            "r3_ordering":s3.mean().item(),
            "r4_nonneg":  s4.mean().item(),
            "r5_species": s5.mean().item(),
            "r6_seasonal":s6.mean().item(),
        }

        sat_loss = self.agg(
            [s1, s2, s3, s4, s5, s6],
            weights=list(self.WEIGHTS.values()),
        )
        return sat_loss, rule_dict