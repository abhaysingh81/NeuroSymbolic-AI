import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class BiomassDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, scaler=None, fit_scaler=True):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # Pivot so each image has all 5 targets in one row
        self.pivot = self.df.pivot_table(
            index=['image_path', 'Pre_GSHH_NDVI', 'Height_Ave_cm', 'Species', 'Sampling_Date'],
            columns='target_name', values='target'
        ).reset_index()

        # Encode species
        self.le = LabelEncoder()
        self.pivot['species_enc'] = self.le.fit_transform(self.pivot['Species'])

        # Tabular features: ndvi, height, species_enc, month
        self.pivot['month'] = pd.to_datetime(self.pivot['Sampling_Date']).dt.month
        self.tab_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'species_enc', 'month']
        self.target_cols = list(self.df['target_name'].unique())  # 5 columns
        self.cont_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm']

        # Scale tabular features
        if scaler is None:
            self.scaler = StandardScaler()
            if fit_scaler:
                self.pivot[self.cont_cols] = self.scaler.fit_transform(self.pivot[self.cont_cols])
        else:
            self.scaler = scaler
            self.pivot[self.cont_cols] = self.scaler.transform(self.pivot[self.cont_cols])

    def __len__(self):
        return len(self.pivot)

    def __getitem__(self, idx):
        row = self.pivot.iloc[idx]
        img = Image.open(f"{self.img_dir}/{row['image_path']}").convert('RGB')
        if self.transform:
            img = self.transform(img)
        tabular = torch.tensor(row[self.tab_cols].values.astype(np.float32))
        targets = torch.tensor(row[self.target_cols].values.astype(np.float32))
        return img, tabular, targets