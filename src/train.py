import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import os

from model import BiomassLTNModel
from dataset import BiomassDataset
from ltn_rules import OrderingRule, NonNegativeRule, HeightBiomassRule, SpeciesConsistencyRule, ltn_sat_loss
from evaluate import evaluate_all

def train(cfg):
    # Transforms
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    full_ds = BiomassDataset('data/train.csv', 'data/images', transform=train_tfm)
    n_val = int(len(full_ds) * 0.2)
    train_ds, val_ds = random_split(full_ds, [len(full_ds)-n_val, n_val])

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=4)

    model  = BiomassLTNModel().cuda()
    rules  = {
        'ordering':    OrderingRule().cuda(),
        'nonneg':      NonNegativeRule().cuda(),
        'height':      HeightBiomassRule().cuda(),
        'species':     SpeciesConsistencyRule(128).cuda(),
    }
    optim  = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=50)
    mse    = nn.MSELoss()

    # Rule weights — tune these in ablation study
    rule_weights = {'ordering': 1.0, 'nonneg': 2.0, 'height': 0.5, 'species': 0.5}
    ltn_lambda = 0.3  # balance between MSE and LTN loss

    # --- TRACKERS ---
    history = {'train_loss': [], 'val_rmse': []}
    best_val_rmse = float('inf')  # Start with infinity for best model tracking

    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        batches = 0
        
        for imgs, tab, targets in train_dl:
            imgs, tab, targets = imgs.cuda(), tab.cuda(), targets.cuda()
            preds, shared_feat = model(imgs, tab)

            # 1) Regression loss
            loss_mse = mse(preds, targets)

            # 2) LTN satisfaction losses
            sats = [
                rules['ordering'](preds),
                rules['nonneg'](preds),
                rules['height'](preds, tab[:, 1]),  # height column
                rules['species'](shared_feat, tab[:, 2].long()),
            ]
            loss_ltn = ltn_sat_loss(sats, list(rule_weights.values()))

            # 3) Combined loss
            loss = loss_mse + ltn_lambda * loss_ltn

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            
            epoch_loss += loss.item()
            batches += 1

        sched.step()
        
        # Log average training loss
        avg_train_loss = epoch_loss / batches
        history['train_loss'].append(avg_train_loss)
        
        print(f"\n--- Epoch {epoch+1}/{cfg['epochs']} Complete ---")
        print(f"Train Loss: {avg_train_loss:.4f}")
        
        target_names = full_ds.target_cols 
        val_results = evaluate_all(model, val_dl, target_names)
        
        # Calculate mean RMSE across all targets
        mean_rmse = sum(metrics['RMSE'] for metrics in val_results.values()) / len(val_results)
        history['val_rmse'].append(mean_rmse)
        
        for target, metrics in val_results.items():
            print(f"{target} - RMSE: {metrics['RMSE']:.4f}, R2: {metrics['R2']:.4f}")

        # --- BEST MODEL CHECKPOINTING ---
        if mean_rmse < best_val_rmse:
            best_val_rmse = mean_rmse
            print(f"*** New best validation RMSE ({best_val_rmse:.4f})! Saving model... ***")
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_biomass_model.pth')

    # --- SAVE PLOTS AT THE END ---
    print("\nTraining finished! Generating plots...")
    os.makedirs('report', exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, cfg['epochs']+1), history['train_loss'], label='Train Loss (MSE + LTN)', color='blue', marker='o', markersize=4)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot 2: Validation RMSE
    plt.subplot(1, 2, 2)
    plt.plot(range(1, cfg['epochs']+1), history['val_rmse'], label='Mean Val RMSE', color='orange', marker='o', markersize=4)
    plt.title('Validation RMSE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average RMSE')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plot_path = 'report/training_curves.png'
    plt.savefig(plot_path)
    print(f"Graph saved successfully to: {plot_path}")
    print(f"Best model saved during training with Validation RMSE: {best_val_rmse:.4f}")
    
    plt.show()

if __name__ == "__main__":
    config = {
        'epochs': 100  
    }
    
    print("Starting training pipeline...")
    train(config)