import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from dataset import DataModule
from model import build_model
from ltn_rules import RuleBank


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(((pred - target) ** 2).mean()).item()


def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def ltn_lambda_schedule(epoch: int, warmup_epochs: int, max_lambda: float) -> float:
    """Linearly ramp LTN weight from 0 to max_lambda over warmup_epochs."""
    if epoch >= warmup_epochs:
        return max_lambda
    return max_lambda * epoch / warmup_epochs


# ─────────────────────────────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, rules, lam, scaler, device, model_type):
    model.train()
    total_loss = total_mse = total_ltn = 0.0
    n_batches = 0

    for imgs, tab, targets in loader:
        imgs    = imgs.to(device, non_blocking=True)
        tab     = tab.to(device,  non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            preds, shared = model(imgs, tab)
            mse_loss = criterion(preds, targets)

            if model_type == "ltn" and rules is not None and lam > 0:
                species_ids = tab[:, 2].long().clamp(0)
                sat_loss, _ = rules(preds, shared, tab, species_ids)
                loss = mse_loss + lam * sat_loss
                total_ltn += sat_loss.item()
            else:
                loss = mse_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_mse  += mse_loss.item()
        n_batches  += 1

    return {
        "loss":     total_loss / n_batches,
        "mse_loss": total_mse  / n_batches,
        "ltn_loss": total_ltn  / n_batches,
    }


# ─────────────────────────────────────────────────────────────────────
# VALIDATE ONE EPOCH
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, criterion, rules, lam, device, model_type, log_targets, rule_bank):
    model.eval()
    all_preds, all_targets = [], []
    total_loss = total_mse = total_ltn = 0.0
    rule_accum = {}
    n_batches = 0

    for imgs, tab, targets in loader:
        imgs    = imgs.to(device, non_blocking=True)
        tab     = tab.to(device,  non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            preds, shared = model(imgs, tab)
            mse_loss = criterion(preds, targets)

            if model_type == "ltn" and rule_bank is not None and lam > 0:
                species_ids = tab[:, 2].long().clamp(0)
                sat_loss, rule_dict = rule_bank(preds, shared, tab, species_ids)
                loss = mse_loss + lam * sat_loss
                total_ltn += sat_loss.item()
                for k, v in rule_dict.items():
                    rule_accum[k] = rule_accum.get(k, 0.0) + v
            else:
                loss = mse_loss

        total_loss += loss.item()
        total_mse  += mse_loss.item()
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        n_batches += 1

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Invert log1p if targets were transformed
    if log_targets:
        all_preds   = torch.expm1(all_preds).clamp(min=0)
        all_targets = torch.expm1(all_targets).clamp(min=0)

    per_target_rmse = {
        f"rmse_{i}": rmse(all_preds[:, i], all_targets[:, i])
        for i in range(all_preds.shape[1])
    }

    avg_rule = {k: v / n_batches for k, v in rule_accum.items()}

    return {
        "loss":     total_loss / n_batches,
        "mse_loss": total_mse  / n_batches,
        "ltn_loss": total_ltn  / n_batches,
        **per_target_rmse,
        **avg_rule,
    }


# ─────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────

def train(cfg: dict):
    # ── Setup ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    run_dir = Path(f"runs/{cfg['model']}_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir()

    writer = SummaryWriter(log_dir=str(run_dir))
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Data ──────────────────────────────────────────────────────────
    dm = DataModule(
        csv_path=cfg["csv_path"],
        img_dir=cfg["img_dir"],
        batch_size=cfg["batch_size"],
        img_size=cfg["img_size"],
        num_workers=cfg.get("num_workers", 4),
        log_targets=cfg.get("log_targets", True),
        random_state=cfg.get("seed", 42),
    )
    dm.setup()
    train_dl, val_dl = dm.get_loaders()
    num_targets = len(dm.target_cols)

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(cfg["model"], num_targets=num_targets).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Trainable parameters: {total_params:,}")

    # ── LTN rules ─────────────────────────────────────────────────────
    rules = RuleBank(p=cfg.get("sat_p", 2.0)).to(device) if cfg["model"] == "ltn" else None

    # ── Optimizer + Scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + (list(rules.parameters()) if rules else []),
        lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4),
    )
    total_steps   = cfg["epochs"] * len(train_dl)
    warmup_steps  = int(0.1 * total_steps)

    criterion = nn.HuberLoss(delta=1.0)  # more robust to outliers than MSE
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    history = []

    # ── Training loop ─────────────────────────────────────────────────
    global_step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        lam = ltn_lambda_schedule(epoch, cfg.get("ltn_warmup_epochs", 10), cfg.get("ltn_lambda", 0.3))

        # Update LR manually for warmup+cosine
        for pg in optimizer.param_groups:
            pg["lr"] = cosine_with_warmup(global_step, warmup_steps, total_steps, cfg["lr"])

        t0 = time.time()
        train_stats = train_epoch(model, train_dl, optimizer, criterion, rules, lam, scaler, device, cfg["model"])
        val_stats   = val_epoch(model, val_dl, criterion, rules, lam, device, cfg["model"], dm.log_targets, rules)

        elapsed = time.time() - t0
        global_step += len(train_dl)

        # ── Logging ───────────────────────────────────────────────────
        writer.add_scalar("Loss/train",     train_stats["loss"],     epoch)
        writer.add_scalar("Loss/val",       val_stats["loss"],       epoch)
        writer.add_scalar("MSE/train",      train_stats["mse_loss"], epoch)
        writer.add_scalar("MSE/val",        val_stats["mse_loss"],   epoch)
        writer.add_scalar("LTN/lambda",     lam,                     epoch)
        if rules:
            writer.add_scalar("LTN/sat_loss_train", train_stats["ltn_loss"], epoch)
            for rule_name, sat_val in val_stats.items():
                if rule_name.startswith("r"):
                    writer.add_scalar(f"RuleSat/{rule_name}", sat_val, epoch)

        rmse_vals = [v for k, v in val_stats.items() if k.startswith("rmse_")]
        mean_rmse = np.mean(rmse_vals) if rmse_vals else float("nan")
        writer.add_scalar("RMSE/val_mean", mean_rmse, epoch)
        for i, v in enumerate(rmse_vals):
            writer.add_scalar(f"RMSE/target_{i}", v, epoch)

        print(
            f"Epoch {epoch:3d}/{cfg['epochs']} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} | "
            f"val_rmse={mean_rmse:.2f} | "
            f"lam={lam:.3f} | {elapsed:.1f}s"
        )

        record = {"epoch": epoch, "lam": lam, **train_stats,
                  **{f"val_{k}": v for k, v in val_stats.items()}}
        history.append(record)

        # ── Checkpoint ────────────────────────────────────────────────
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state":optimizer.state_dict(),
                "val_loss":       best_val_loss,
                "target_cols":    dm.target_cols,
                "scaler_mean":    dm.scaler.mean_.tolist() if dm.scaler else None,
                "scaler_scale":   dm.scaler.scale_.tolist() if dm.scaler else None,
            }, ckpt_dir / "best.pt")

    writer.close()

    # ── Save history ──────────────────────────────────────────────────
    import pandas as pd
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Run saved to: {run_dir}")
    return run_dir


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train biomass prediction model")
    p.add_argument("--model",        type=str,   default="ltn",       choices=["ltn","neural","symbolic"])
    p.add_argument("--csv_path",     type=str,   default="data/train.csv")
    p.add_argument("--img_dir",      type=str,   default="data/images")
    p.add_argument("--epochs",       type=int,   default=60)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--img_size",     type=int,   default=224)
    p.add_argument("--ltn_lambda",   type=float, default=0.3)
    p.add_argument("--ltn_warmup",   type=int,   default=10)
    p.add_argument("--sat_p",        type=float, default=2.0)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--no_log_targets", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = vars(args)
    cfg["log_targets"]       = not cfg.pop("no_log_targets")
    cfg["ltn_warmup_epochs"] = cfg.pop("ltn_warmup")
    train(cfg)