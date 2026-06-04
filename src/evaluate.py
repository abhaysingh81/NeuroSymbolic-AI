import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error
from torchvision import transforms

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#FAFAFA",
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, targets: np.ndarray, target_names: List[str]) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(target_names):
        p = preds[:, i]; t = targets[:, i]
        rows.append({
            "target":  name,
            "RMSE":    np.sqrt(np.mean((p - t) ** 2)),
            "MAE":     mean_absolute_error(t, p),
            "R2":      r2_score(t, p),
            "MAPE":    np.mean(np.abs((p - t) / (t + 1e-6))) * 100,
        })
    return pd.DataFrame(rows).set_index("target")


# ─────────────────────────────────────────────────────────────────────
# LOAD MODEL + RUN INFERENCE
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: nn.Module, loader, device: torch.device, log_targets: bool = True):
    model.eval()
    all_preds, all_targets, all_shared = [], [], []

    for imgs, tab, targets in loader:
        imgs    = imgs.to(device)
        tab     = tab.to(device)
        preds, shared = model(imgs, tab)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_shared.append(shared.cpu())

    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    shared  = torch.cat(all_shared).numpy()

    if log_targets:
        preds   = np.expm1(np.clip(preds, -10, 20))
        targets = np.expm1(np.clip(targets, -10, 20))

    return preds, targets, shared


# ─────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────

def print_comparison_table(results: Dict[str, pd.DataFrame]):
    print("\n" + "="*75)
    print(f"{'MODEL COMPARISON':^75}")
    print("="*75)

    metric_dfs = {}
    for metric in ["RMSE", "R2", "MAE"]:
        rows = {}
        for model_name, df in results.items():
            rows[model_name] = df[metric]
        metric_dfs[metric] = pd.DataFrame(rows)
        print(f"\n── {metric} ─────────────────────────")
        print(metric_dfs[metric].round(3).to_string())

    # Mean across targets
    print("\n── MEAN ACROSS ALL TARGETS ──────────")
    summary = pd.DataFrame({
        model: {
            "mean RMSE": df["RMSE"].mean(),
            "mean MAE":  df["MAE"].mean(),
            "mean R²":   df["R2"].mean(),
        }
        for model, df in results.items()
    }).T.round(3)
    print(summary.to_string())
    print("="*75)
    return summary


# ─────────────────────────────────────────────────────────────────────
# SCATTER PLOTS: predicted vs actual
# ─────────────────────────────────────────────────────────────────────

def plot_scatter(results_preds: Dict, target_names: List[str], save_dir: str):
    n = len(target_names)
    n_models = len(results_preds)
    colors = ["#5B8DB8", "#E07B54", "#6BAF74"]

    fig, axes = plt.subplots(n_models, n, figsize=(4.5*n, 4*n_models))
    if n_models == 1: axes = axes[None, :]

    for mi, (model_name, (preds, targets)) in enumerate(results_preds.items()):
        for ti, tgt in enumerate(target_names):
            ax = axes[mi, ti]
            p = preds[:, ti]; t = targets[:, ti]
            ax.scatter(t, p, alpha=0.25, s=10, color=colors[mi % len(colors)])
            lim = max(t.max(), p.max()) * 1.05
            ax.plot([0, lim], [0, lim], "k--", lw=1.2)
            r2 = r2_score(t, p)
            ax.set_xlabel("Actual (g)"); ax.set_ylabel("Predicted (g)")
            ax.set_title(f"{model_name}\n{tgt} | R²={r2:.3f}", fontsize=9)

    plt.suptitle("Predicted vs Actual Biomass per Model & Target", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/scatter_predicted_actual.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: scatter_predicted_actual.png")


# ─────────────────────────────────────────────────────────────────────
# RULE SATISFACTION EVOLUTION (from training history)
# ─────────────────────────────────────────────────────────────────────

def plot_rule_satisfaction(history_csv: str, save_dir: str):
    df = pd.read_csv(history_csv)
    rule_cols = [c for c in df.columns if c.startswith("val_r") and "rule" not in c.lower()]

    if not rule_cols:
        print("No rule satisfaction columns found in history. Skipping plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    colors = plt.cm.tab10.colors

    for i, (col, ax) in enumerate(zip(rule_cols, axes)):
        rule_label = col.replace("val_", "").replace("_", " ").title()
        ax.plot(df["epoch"], df[col], color=colors[i], lw=2)
        ax.fill_between(df["epoch"], df[col], alpha=0.15, color=colors[i])
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_title(rule_label, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Satisfaction")

    for ax in axes[len(rule_cols):]:
        ax.set_visible(False)

    plt.suptitle("LTN Rule Satisfaction Over Training (higher = better)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/rule_satisfaction_curves.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: rule_satisfaction_curves.png")


# ─────────────────────────────────────────────────────────────────────
# ABLATION: LTN vs LTN-without-each-rule
# ─────────────────────────────────────────────────────────────────────

def plot_ablation_bar(ablation_results: Dict[str, float], save_dir: str):
    """
    ablation_results: {"Full LTN": mean_rmse, "w/o R1_NDVI": ..., ...}
    """
    labels = list(ablation_results.keys())
    values = list(ablation_results.values())

    # Highlight the full model
    colors = ["#E07B54" if "Full" not in l else "#5B8DB8" for l in labels]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55)
    ax.set_xlabel("Mean RMSE (g)")
    ax.set_title("Ablation Study: Impact of Each LTN Rule on Validation RMSE",
                 fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9)

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ablation_bar.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: ablation_bar.png")


# ─────────────────────────────────────────────────────────────────────
# SHAP FEATURE IMPORTANCE (tabular branch only)
# ─────────────────────────────────────────────────────────────────────

def plot_shap_tabular(model: nn.Module, val_tab: np.ndarray,
                      feature_names: List[str], save_dir: str, device):
    """
    Uses gradient-based approximation of SHAP values for the tabular encoder.
    Requires shap package (pip install shap).
    """
    try:
        import shap
    except ImportError:
        print("shap not installed. Skipping SHAP plot. (pip install shap)")
        return

    model.eval()
    tab_tensor = torch.tensor(val_tab[:200], dtype=torch.float32).to(device)

    def tabular_predict(x):
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).to(device)
            # Use a dummy image of zeros
            dummy_img = torch.zeros(t.shape[0], 3, 224, 224, device=device)
            preds, _ = model(dummy_img, t)
            return preds.cpu().numpy()

    background = tab_tensor[:50].cpu().numpy()
    explainer  = shap.KernelExplainer(tabular_predict, background)
    shap_vals  = explainer.shap_values(tab_tensor[:100].cpu().numpy(), nsamples=50)

    # Mean absolute SHAP per feature across all targets
    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#5B8DB8" if v == max(mean_shap) else "#B5C9D5" for v in mean_shap]
    ax.barh(feature_names, mean_shap, color=colors, edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Tabular Feature Importance via SHAP", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_tabular.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: shap_tabular.png")


# ─────────────────────────────────────────────────────────────────────
# TRAINING CURVE COMPARISON
# ─────────────────────────────────────────────────────────────────────

def plot_training_curves(run_dirs: Dict[str, str], save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"ltn": "#5B8DB8", "neural": "#E07B54", "symbolic": "#6BAF74"}

    for model_name, run_dir in run_dirs.items():
        hist_path = Path(run_dir) / "history.csv"
        if not hist_path.exists():
            continue
        df   = pd.read_csv(hist_path)
        col  = colors.get(model_name, "gray")
        label = model_name.upper()

        axes[0].plot(df["epoch"], df["loss"],       color=col, lw=2, label=f"{label} train")
        axes[0].plot(df["epoch"], df["val_loss"],   color=col, lw=2, ls="--", alpha=0.7)
        axes[1].plot(df["epoch"], df.filter(like="rmse").mean(axis=1), color=col, lw=2, label=label)

    for ax, title in zip(axes, ["Total Loss (solid=train, dashed=val)", "Mean Val RMSE (g)"]):
        ax.set_xlabel("Epoch"); ax.legend(fontsize=9)
        ax.set_title(title, fontweight="bold")

    plt.suptitle("Training Curves — All Models", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: training_curves.png")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ltn_run",      type=str, required=True,  help="Path to LTN run directory")
    p.add_argument("--neural_run",   type=str, default=None)
    p.add_argument("--symbolic_run", type=str, default=None)
    p.add_argument("--csv_path",     type=str, default="data/train.csv")
    p.add_argument("--img_dir",      type=str, default="data/images")
    p.add_argument("--save_dir",     type=str, default="report/figures")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dirs = {"ltn": args.ltn_run}
    if args.neural_run:   run_dirs["neural"]   = args.neural_run
    if args.symbolic_run: run_dirs["symbolic"] = args.symbolic_run

    # Plot training curves if all runs given
    plot_training_curves(run_dirs, args.save_dir)

    # Plot rule satisfaction for LTN
    ltn_hist = Path(args.ltn_run) / "history.csv"
    if ltn_hist.exists():
        plot_rule_satisfaction(str(ltn_hist), args.save_dir)