# Pasture Biomass Prediction with Neuro-Symbolic AI

Predicting five pasture biomass components from drone/field imagery and agronomic metadata, using Logical Tensor Networks (LTN) to embed domain rules directly into training.

---

## What this project does

Standard neural networks predict biomass decently but nothing stops them from producing physically impossible outputs — like negative grass weight, or predicting more green leaf than total plant mass. This project solves that by grounding domain knowledge as differentiable constraints that train alongside the network.

The model fuses EfficientNet-B2 image features with tabular data (NDVI, canopy height, species, sampling date) through cross-modal attention, then enforces six agronomic rules during training using fuzzy logic. The result is a model that's both accurate *and* physically consistent — and one you can interrogate to understand why it makes each prediction.

---

## Dataset

Two files are expected under `data/`:

```
data/
  train.csv       # metadata + biomass labels
  images/         # pasture .jpg images (referenced in train.csv)
```

`train.csv` has one row per biomass component per image:

| Column | Description |
|--------|-------------|
| `image_path` | relative path to image |
| `target_name` | which biomass component (5 unique values) |
| `target` | biomass in grams |
| `Pre_GSHH_NDVI` | Normalised Difference Vegetation Index |
| `Height_Ave_cm` | canopy height in cm |
| `Species` | pasture species name |
| `Sampling_Date` | sampling date |

---

## Project structure

```
project/
├── data/
│   ├── train.csv
│   └── images/
├── src/
│   ├── dataset.py        # data loading, augmentation, train/val split
│   ├── model.py          # all three model variants
│   ├── ltn_rules.py      # LTN predicates, fuzzy operators, SatAgg
│   ├── train.py          # training loop (all models)
│   └── evaluate.py       # metrics, plots, SHAP, comparison table
├── notebooks/
│   └── EDA.py            # full exploratory analysis (10 figures)
├── report/
│   └── figures/          # EDA and evaluation plots
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/abhaysingh81/NeuroSymbolic-AI
cd NeuroSymbolic-AI
pip install -r requirements.txt
```

Python 3.10+ recommended. A CUDA-capable GPU cuts training time from ~4 hours to ~25 minutes.

---

## Running the code

### Step 1: Exploratory data analysis

```bash
python notebooks/EDA.py
```

Produces 10 figures in `report/figures/` covering target distributions, NDVI/height correlations, species breakdown, seasonality, and rule-violation analysis.

### Step 2: Train all three models

```bash
# Primary model (LTN neuro-symbolic)
python src/train.py --model ltn --epochs 60 --lr 1e-4 --ltn_lambda 0.3

# Neural baseline (no rules)
python src/train.py --model neural --epochs 60 --lr 1e-4

# Symbolic baseline (no images)
python src/train.py --model symbolic --epochs 40 --lr 5e-3
```

Checkpoints and training history are saved to `runs/<model>_<timestamp>/`.

### Step 3: Monitor training

```bash
tensorboard --logdir runs/
```

### Step 4: Evaluate and compare

```bash
python src/evaluate.py \
  --ltn_run      runs/ltn_<timestamp> \
  --neural_run   runs/neural_<timestamp> \
  --symbolic_run runs/symbolic_<timestamp>
```

---

## The three models

| Model | What it does |
|-------|-------------|
| **Neural baseline** | EfficientNet + tabular MLP, concat fusion, MSE loss. No domain rules. |
| **Symbolic baseline** | Tabular only. Hand-crafted NDVI×height interactions, species one-hot, linear regression on top. No images. |
| **LTN ** | Full multi-modal with cross-attention fusion. Six agronomic rules embedded as fuzzy-logic soft constraints. |

---

## The LTN rules

Rules are encoded as differentiable functions returning satisfaction scores in [0, 1]. They are aggregated with the pMean operator and added to the regression loss:

```
L_total = L_Huber(pred, target) + λ · L_sat(rules)
```

The six rules:

| Rule | Agronomic basis | EDA finding |
|------|-----------------|-------------|
| R1: High NDVI → High biomass | NDVI measures photosynthetic activity | r > 0.4 across all targets |
| R2: Height ↑ → Biomass ↑ | Canopy architecture correlates with yield | r ≈ 0.35–0.6 by species |
| R3: Total ≥ each component | Set containment — total is sum of parts | < 3% violation in training data |
| R4: All predictions ≥ 0 | Biomass is a physical mass | Hard physical constraint |
| R5: Species consistency | Same species → similar embedding | Intra-species variance < inter-species |
| R6: Seasonal correlation | Spring/summer growth peaks | NDVI peaks align with biomass peaks |

λ ramps linearly from 0 to 0.3 over the first 10 epochs so the model learns basic regression before constraints are applied.

---

## Key implementation decisions

**EfficientNet-B2 over ResNet-18** — better feature extraction for vegetation texture, roughly 10% RMSE improvement in preliminary tests.

**log1p target transformation** — biomass distributions are right-skewed (skewness 1.8–3.2). Log-transforming targets stabilises gradients and reduces sensitivity to large outliers.

**Cross-modal attention** — rather than naively concatenating image and tabular features, the image embedding queries the tabular context. This lets the model weight image regions differently depending on NDVI or species.

**Group-stratified split** — the train/val split groups by the first path component of `image_path` so images from the same pasture/date always stay in the same split. This prevents temporal data leakage.

**HuberLoss over MSE** — biomass has heavy-tailed outliers. Huber loss (δ=1.0) reduces their influence without throwing them away entirely.

---

## Reproducing the reported results

```bash
# Seed is fixed to 42 in all runs
python src/train.py --model ltn     --seed 42 --epochs 60
python src/train.py --model neural  --seed 42 --epochs 60
python src/train.py --model symbolic --seed 42 --epochs 40
```

---

## What the rule satisfaction curves tell you

Each LTN rule's satisfaction score is logged every epoch. A well-trained model should show:
- R4 (non-negativity) converging to ~0.99 quickly — easy constraint
- R3 (ordering) stabilising around 0.90–0.95
- R1/R2 (NDVI and height correlations) gradually improving as the model learns the regression signal

If R5 (species consistency) stays low, consider increasing its weight or adding more species-stratified augmentation.

---



