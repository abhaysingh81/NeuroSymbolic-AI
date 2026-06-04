from sklearn.metrics import r2_score
import numpy as np
import torch

def evaluate_all(model, val_dl, target_names):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, tab, targets in val_dl:
            preds, _ = model(imgs.cuda(), tab.cuda())
            all_preds.append(preds.cpu())
            all_targets.append(targets)

    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    results = {}
    for i, name in enumerate(target_names):
        mse  = np.mean((preds[:,i] - targets[:,i])**2)
        rmse = np.sqrt(mse)
        r2   = r2_score(targets[:,i], preds[:,i])
        mae  = np.mean(np.abs(preds[:,i] - targets[:,i]))
        results[name] = {'RMSE': rmse, 'R2': r2, 'MAE': mae}
    return results