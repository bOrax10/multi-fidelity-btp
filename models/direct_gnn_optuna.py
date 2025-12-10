#!/usr/bin/env python3

import os
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import optuna
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.nn import Linear, Dropout, ReLU
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import SchNet

# -----------------------
# Config / Constant Setup
# -----------------------
SEED = 42
INPUT_DATA_FILE = "../../data/interim/urea.parquet"
TARGET_COLUMN = "def2-TZVP_TrDP_1_x"
OUTPUT_FIG_PATH = f"../../reports/figures/gnn_optuna_{TARGET_COLUMN}.png"
LOSS_FIG_PATH = f"../../reports/figures/gnn_loss_curve_{TARGET_COLUMN}.png"

N_TRIALS = 30
TIMEOUT = 60 * 60 * 2
PRUNING = True

GRAD_CLIP = 5.0
TRAIN_VAL_SPLIT = 0.8
NUM_GAUSSIANS = 50

LOW_FID_FEATURE_COLUMNS = [
    "STO-3G_TrDP_1_x",
    "3-21G_TrDP_1_x",
    "6-31G_TrDP_1_x",
    "def2-SVP_TrDP_1_x",
]


# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

# -----------------------
# Helper Functions
# -----------------------
atomic_symbol_to_number = {"H": 1, "C": 6, "N": 7, "O": 8}


def parse_atom_info(df: pd.DataFrame):
    coord_pattern = re.compile(r"^[A-Za-z]+_\d+_[xyz]$")
    all_coord_cols = [col for col in df.columns if coord_pattern.match(col)]
    if len(all_coord_cols) == 0:
        raise ValueError("No coordinate columns found.")
    atom_labels = sorted(list(set([col[:-2] for col in all_coord_cols])))
    atomic_numbers = []
    for label in atom_labels:
        symbol = re.match(r"([A-Za-z]+)_", label).group(1)
        atomic_numbers.append(atomic_symbol_to_number[symbol])
    return torch.tensor(atomic_numbers, dtype=torch.long), atom_labels


def create_graph_dataset(
    df: pd.DataFrame, target_column: str, atom_labels=None, extra_features=None
):
    if atom_labels is None:
        _, atom_labels = parse_atom_info(df)
    atomic_numbers_tensor, _ = parse_atom_info(df)

    all_coord_cols_flat = [
        f"{label}_{dim}" for label in atom_labels for dim in ["x", "y", "z"]
    ]
    coords_flat = df[all_coord_cols_flat].values
    n_frames = len(df)
    n_atoms = len(atom_labels)
    all_pos_tensor = torch.tensor(
        coords_flat.reshape(n_frames, n_atoms, 3), dtype=torch.float
    )
    y_tensor = torch.tensor(df[target_column].values, dtype=torch.float)

    if extra_features is not None:
        extra_tensor = torch.tensor(extra_features, dtype=torch.float)
    else:
        extra_tensor = None

    graph_list = []
    for i in range(n_frames):
        kwargs = dict(
            x=atomic_numbers_tensor.clone(),
            pos=all_pos_tensor[i].clone(),
            y=y_tensor[i].clone().unsqueeze(0),
        )
        if extra_tensor is not None:
            kwargs["extra"] = extra_tensor[i].unsqueeze(0).clone()

        graph_list.append(Data(**kwargs))
    return graph_list


# -----------------------
# Model Definition
# -----------------------
class GNNModel(torch.nn.Module):
    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        cutoff=5.0,
        num_gaussians=NUM_GAUSSIANS,
        dropout=0.1,
        num_extra: int = 0,
    ):
        super(GNNModel, self).__init__()
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
        )
        self.num_extra = num_extra
        in_dim = 1 + (num_extra if num_extra > 0 else 0)

        self.head = torch.nn.Sequential(
            Linear(in_dim, hidden_channels // 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_channels // 2, 1),
        )

    def forward(self, data):
        z = data.x.long()
        pos = data.pos
        batch = data.batch

        sch_out = self.schnet(z, pos, batch)

        if self.num_extra > 0 and hasattr(data, "extra") and data.extra is not None:
            extra = data.extra
            if extra.dim() == 1:
                extra = extra.unsqueeze(1)
            sch_out = torch.cat([sch_out, extra.to(sch_out.device)], dim=1)

        return self.head(sch_out).squeeze(1)


# -----------------------
# Data Preparation
# -----------------------
def load_and_prepare_data(input_file, target_col):
    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)

    group_col = next(
        (c for c in ["molecule_id", "mol_id", "traj_id"] if c in df.columns), None
    )
    if group_col:
        print(f"Splitting by {group_col}")
        unique_groups = df[group_col].unique()
        np.random.shuffle(unique_groups)
        n_train = int(TRAIN_VAL_SPLIT * len(unique_groups))
        train_groups = unique_groups[:n_train]
        val_groups = unique_groups[n_train:]
        train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
        val_df = df[df[group_col].isin(val_groups)].reset_index(drop=True)
    else:
        print("Random split")
        idx = np.arange(len(df))
        np.random.shuffle(idx)
        n_train = int(TRAIN_VAL_SPLIT * len(idx))
        train_df = df.iloc[idx[:n_train]].reset_index(drop=True)
        val_df = df.iloc[idx[n_train:]].reset_index(drop=True)

    y_scaler = StandardScaler()
    y_scaler.fit(train_df[[target_col]].values)
    train_df["_y_scaled"] = y_scaler.transform(train_df[[target_col]].values).flatten()
    val_df["_y_scaled"] = y_scaler.transform(val_df[[target_col]].values).flatten()

    available_lf_cols = [c for c in LOW_FID_FEATURE_COLUMNS if c in df.columns]
    num_extra = len(available_lf_cols)
    lf_scaler = None
    train_extra, val_extra = None, None

    if num_extra > 0:
        lf_scaler = StandardScaler()
        lf_scaler.fit(train_df[available_lf_cols].values)
        train_extra = lf_scaler.transform(train_df[available_lf_cols].values)
        val_extra = lf_scaler.transform(val_df[available_lf_cols].values)

    _, atom_labels = parse_atom_info(train_df)
    train_dataset = create_graph_dataset(
        train_df, "_y_scaled", atom_labels, train_extra
    )
    val_dataset = create_graph_dataset(val_df, "_y_scaled", atom_labels, val_extra)

    return train_dataset, val_dataset, y_scaler, lf_scaler, num_extra


# -----------------------
# Optuna Objective
# -----------------------
def objective(trial, train_dataset, val_dataset, num_extra, device):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    num_interactions = trial.suggest_int("num_interactions", 3, 6)
    cutoff = trial.suggest_float("cutoff", 4.0, 8.0)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GNNModel(
        hidden_channels=hidden_channels,
        num_filters=hidden_channels,
        num_interactions=num_interactions,
        cutoff=cutoff,
        dropout=dropout,
        num_extra=num_extra,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.SmoothL1Loss()

    tuning_epochs = 50

    for epoch in range(tuning_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y.view(-1).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        model.eval()
        total_val_loss = 0.0
        total_graphs = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = loss_fn(out, batch.y.view(-1).to(device))
                total_val_loss += loss.item() * batch.num_graphs
                total_graphs += batch.num_graphs

        avg_val_loss = total_val_loss / total_graphs
        trial.report(avg_val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val_loss


# -----------------------
# Final Training Run
# -----------------------
def run_final_training(
    best_params,
    train_dataset,
    val_dataset,
    y_scaler,
    lf_scaler,
    num_extra,
    device,
):
    print("\nRunning Final Training with Best Params...")
    epochs = 150
    batch_size = best_params["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GNNModel(
        hidden_channels=best_params["hidden_channels"],
        num_filters=best_params["hidden_channels"],
        num_interactions=best_params["num_interactions"],
        cutoff=best_params["cutoff"],
        dropout=best_params["dropout"],
        num_extra=num_extra,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=best_params["lr"], weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    loss_fn = torch.nn.SmoothL1Loss()

    ckpt_path = Path("checkpoints") / f"best_optuna_model_{TARGET_COLUMN}.pt"
    ckpt_path.parent.mkdir(exist_ok=True)

    best_val_loss = float("inf")
    train_hist = []
    val_hist = []
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_count = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y.view(-1).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            t_loss += loss.item() * batch.num_graphs
            t_count += batch.num_graphs

        model.eval()
        v_loss, v_count = 0.0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = loss_fn(out, batch.y.view(-1).to(device))
                v_loss += loss.item() * batch.num_graphs
                v_count += batch.num_graphs
                all_preds.append(out.cpu())
                all_targets.append(batch.y.view(-1).cpu())

        avg_t = t_loss / t_count
        avg_v = v_loss / v_count

        # Store for plotting
        train_hist.append(avg_t)
        val_hist.append(avg_v)

        scheduler.step(avg_v)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Ep {epoch} | Train: {avg_t:.5f} | Val: {avg_v:.5f}")

        if avg_v < best_val_loss:
            best_val_loss = avg_v
            patience_cnt = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_params": best_params,
                    "lf_scaler": lf_scaler,
                    "y_scaler": y_scaler,
                },
                str(ckpt_path),
            )
        else:
            patience_cnt += 1
            if patience_cnt >= 25:
                print("Early stopping.")
                break

    # Learning Curve Plot
    Path(os.path.dirname(LOSS_FIG_PATH) or ".").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_hist, label="Train Loss")
    plt.plot(val_hist, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (SmoothL1)")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(LOSS_FIG_PATH)
    print(f"Learning curve saved to {LOSS_FIG_PATH}")

    # Scatter Plot Evaluation
    all_preds = torch.cat(all_preds).numpy().reshape(-1, 1)
    all_targets = torch.cat(all_targets).numpy().reshape(-1, 1)
    preds_orig = y_scaler.inverse_transform(all_preds).flatten()
    targets_orig = y_scaler.inverse_transform(all_targets).flatten()

    mae = mean_absolute_error(targets_orig, preds_orig)
    r2 = r2_score(targets_orig, preds_orig)
    print(f"\nFinal Result -> MAE: {mae:.6f} | R2: {r2:.6f}")

    Path(os.path.dirname(OUTPUT_FIG_PATH) or ".").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=targets_orig, y=preds_orig, alpha=0.6)
    mn, mx = min(targets_orig.min(), preds_orig.min()), max(
        targets_orig.max(), preds_orig.max()
    )
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.title(f"Optuna Best Model: MAE={mae:.4f} | R2={r2:.4f}")
    plt.savefig(OUTPUT_FIG_PATH)
    print(f"Scatter plot saved to {OUTPUT_FIG_PATH}")


# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, y_scaler, lf_scaler, n_extra = load_and_prepare_data(
        INPUT_DATA_FILE, TARGET_COLUMN
    )

    print(f"\nStarting Optuna Optimization ({N_TRIALS} trials)...")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", pruner=pruner)

    func = lambda trial: objective(trial, train_ds, val_ds, n_extra, device)
    study.optimize(func, n_trials=N_TRIALS, timeout=TIMEOUT)

    print("\nBest Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    run_final_training(
        study.best_params,
        train_ds,
        val_ds,
        y_scaler,
        lf_scaler,
        n_extra,
        device,
    )
