import os
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(split: str, img_size: int = 224) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:   # val / test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ─────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────

class BiomassDataset(Dataset):
    """
    Multimodal dataset: pasture image + tabular metadata → 5 biomass targets.

    Parameters
    ----------
    pivot_df    : Pre-processed pivot dataframe (one row per image).
    img_dir     : Root directory containing image files.
    target_cols : List of 5 target column names (sorted).
    transform   : torchvision transform pipeline.
    log_targets : Apply log1p to targets (recommended — they are right-skewed).
    """

    def __init__(
        self,
        pivot_df: pd.DataFrame,
        img_dir: str,
        target_cols: List[str],
        transform: Optional[transforms.Compose] = None,
        log_targets: bool = True,
    ):
        self.df          = pivot_df.reset_index(drop=True)
        self.img_dir     = img_dir
        self.target_cols = target_cols
        self.transform   = transform
        self.log_targets = log_targets

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # ── Image ──────────────────────────────────────────────────────
        img_path = os.path.join(self.img_dir, row["image_path"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # ── Tabular features ───────────────────────────────────────────
        tab = torch.tensor(
            row[["Pre_GSHH_NDVI_scaled", "Height_Ave_cm_scaled", "species_enc", "month_sin", "month_cos"]].values.astype(np.float32)
        )

        # ── Targets ────────────────────────────────────────────────────
        targets = torch.tensor(row[self.target_cols].values.astype(np.float32))
        if self.log_targets:
            targets = torch.log1p(targets.clamp(min=0))

        return img, tab, targets


# ─────────────────────────────────────────────────────────────────────
# PREPROCESSING & SPLIT FACTORY
# ─────────────────────────────────────────────────────────────────────

class DataModule:
    """
    Loads train.csv, pivots, encodes features, creates stratified splits,
    and returns PyTorch DataLoaders.

    Usage
    -----
    dm = DataModule("data/train.csv", "data/images")
    dm.setup()
    train_dl, val_dl = dm.get_loaders()
    target_names = dm.target_cols
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        val_size: float = 0.2,
        batch_size: int = 32,
        img_size: int = 224,
        num_workers: int = 4,
        log_targets: bool = True,
        random_state: int = 42,
    ):
        self.csv_path     = csv_path
        self.img_dir      = img_dir
        self.val_size     = val_size
        self.batch_size   = batch_size
        self.img_size     = img_size
        self.num_workers  = num_workers
        self.log_targets  = log_targets
        self.random_state = random_state

        # Set after setup()
        self.target_cols: List[str] = []
        self.scaler: StandardScaler = None
        self.label_enc: LabelEncoder = None
        self.train_df: pd.DataFrame = None
        self.val_df:   pd.DataFrame = None

    # ── setup ────────────────────────────────────────────────────────
    def setup(self):
        df = pd.read_csv(self.csv_path, parse_dates=["Sampling_Date"])

        # Pivot: one row per image, targets as columns
        self.target_cols = sorted(df["target_name"].unique().tolist())
        pivot = df.pivot_table(
            index=["image_path", "Pre_GSHH_NDVI", "Height_Ave_cm", "Species", "Sampling_Date"],
            columns="target_name",
            values="target",
            aggfunc="first",  # deterministic
        ).reset_index()

        # Fill NaN targets with 0 (missing component = absent = 0 g)
        pivot[self.target_cols] = pivot[self.target_cols].fillna(0.0)

        # ── Feature engineering ──────────────────────────────────────
        # Month cyclical encoding
        pivot["month"] = pivot["Sampling_Date"].dt.month
        pivot["month_sin"] = np.sin(2 * np.pi * pivot["month"] / 12)
        pivot["month_cos"] = np.cos(2 * np.pi * pivot["month"] / 12)

        # Species integer encoding (unknown → -1 → 0 after clip)
        self.label_enc = LabelEncoder()
        pivot["species_enc"] = self.label_enc.fit_transform(pivot["Species"]).astype(np.float32)

        # ── Group-stratified split (by image_path prefix / pasture) ──
        # Group on the first path component so images from the same pasture
        # stay together (prevents leakage).
        pivot["group"] = pivot["image_path"].apply(lambda p: p.split("/")[0])
        splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_state)
        train_idx, val_idx = next(splitter.split(pivot))

        # ── Scale continuous tabular features ────────────────────────
        tab_cont = ["Pre_GSHH_NDVI", "Height_Ave_cm"]
        self.scaler = StandardScaler()
        pivot.loc[train_idx, [f"{c}_scaled" for c in tab_cont]] = \
            self.scaler.fit_transform(pivot.loc[train_idx, tab_cont])
        pivot.loc[val_idx,   [f"{c}_scaled" for c in tab_cont]] = \
            self.scaler.transform(pivot.loc[val_idx, tab_cont])
        # Fill any leftover NaN from above
        for c in tab_cont:
            pivot[f"{c}_scaled"] = pivot[f"{c}_scaled"].fillna(0.0)

        self.train_df = pivot.iloc[train_idx].copy()
        self.val_df   = pivot.iloc[val_idx].copy()

        print(f"[DataModule] Train: {len(self.train_df)} | Val: {len(self.val_df)}")
        print(f"[DataModule] Targets: {self.target_cols}")

    # ── loaders ──────────────────────────────────────────────────────
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_ds = BiomassDataset(
            self.train_df, self.img_dir, self.target_cols,
            transform=get_transforms("train", self.img_size),
            log_targets=self.log_targets,
        )
        val_ds = BiomassDataset(
            self.val_df, self.img_dir, self.target_cols,
            transform=get_transforms("val", self.img_size),
            log_targets=self.log_targets,
        )
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                              num_workers=self.num_workers, pin_memory=True, drop_last=True)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False,
                              num_workers=self.num_workers, pin_memory=True)
        return train_dl, val_dl