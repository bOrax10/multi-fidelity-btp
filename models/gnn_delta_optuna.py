import os
import random
import re
import warnings
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.preprocessing import StandardScaler
from torch.nn import Linear, Dropout, ReLU
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import SchNet

# -----------------------
# Config / HPO Settings
# -----------------------
SEED = 42
DATA_PATH = "../../data/interim/urea.parquet"
TARGET = "def2-TZVP_TrDP_1_x"
BASELINE = "def2-SVP_TrDP_1_x"
OUT_DIR = "reports/figures/optuna_schnet"

BATCH_SIZE = 64
TRIALS = 50
EPOCHS = 50
PATIENCE = 10
SPLIT = 0.75
GRAD_CLIP = 5.0


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
ATOM_MAP = {"H": 1, "C": 6, "N": 7, "O": 8}


def _get_atoms(df: pd.DataFrame):
    pat = re.compile(r"^[A-Za-z]+_\d+_[xyz]$")
    cols = [c for c in df.columns if pat.match(c)]
    if not cols:
        raise ValueError("No coordinate columns found.")
    labels = sorted(list(set([c[:-2] for c in cols])))
    nums = []
    for lbl in labels:
        sym = re.match(r"([A-Za-z]+)_", lbl).group(1)
        nums.append(ATOM_MAP[sym])
    return torch.tensor(nums, dtype=torch.long), labels


def make_graphs(df: pd.DataFrame, target_col: str, labels=None):
    if labels is None:
        _, labels = _get_atoms(df)
    z_tensor, _ = _get_atoms(df)

    cols_flat = [f"{l}_{d}" for l in labels for d in ["x", "y", "z"]]
    coords = df[cols_flat].values
    n_frames = len(df)
    n_atoms = len(labels)

    pos = torch.tensor(coords.reshape(n_frames, n_atoms, 3), dtype=torch.float)
    y = torch.tensor(df[target_col].values, dtype=torch.float)

    graphs = []
    for i in range(n_frames):
        graphs.append(
            Data(x=z_tensor.clone(), pos=pos[i].clone(), y=y[i].clone().unsqueeze(0))
        )
    return graphs


# -----------------------
# Model Definition
# -----------------------
class GNNModel(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_filters: int,
        num_interactions: int,
        num_gaussians: int,
        cutoff: float,
        dropout: float,
    ):
        super(GNNModel, self).__init__()
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
        )
        self.head = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_channels // 2, 1),
        )

    def forward(self, data):
        z = data.x.long()
        pos = data.pos
        batch = data.batch
        out = self.schnet(z, pos, batch)

        if out.dim() == 2 and out.size(1) == 1:
            return out.squeeze(1)
        if out.dim() == 2:
            return self.head(out).squeeze(1)
        return self.head(out.unsqueeze(0)).squeeze(1)


# -----------------------
# Optuna Objective
# -----------------------
def objective(trial, train_df, val_df, labels):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1e-4, log=True)
    hidden = trial.suggest_categorical("hidden", [64, 128, 256])
    interactions = trial.suggest_int("interactions", 3, 8)
    gaussians = trial.suggest_categorical("gaussians", [50, 80])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    cutoff = trial.suggest_categorical("cutoff", [3.0, 5.0, 7.0])

    train_ds = make_graphs(train_df, "_delta_scaled", labels=labels)
    val_ds = make_graphs(val_df, "_delta_scaled", labels=labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GNNModel(
        hidden_channels=hidden,
        num_filters=hidden,
        num_interactions=interactions,
        num_gaussians=gaussians,
        cutoff=cutoff,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    loss_fn = torch.nn.SmoothL1Loss()

    best_val = float("inf")
    patience_cnt = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            tgt = batch.y.view(-1).to(device)
            loss = loss_fn(out, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            t_loss += loss.item() * batch.num_graphs

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                tgt = batch.y.view(-1).to(device)
                loss = loss_fn(out, tgt)
                v_loss += loss.item() * batch.num_graphs

        avg_v = v_loss / len(val_loader.dataset)
        scheduler.step(avg_v)

        trial.report(avg_v, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if avg_v < best_val:
            best_val = avg_v
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= PATIENCE:
            break

    return best_val


# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    print(f"Loading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Rows: {len(df)}")

    grp_col = None
    for cand in ("molecule_id", "mol_id", "traj_id", "trajectory_id"):
        if cand in df.columns:
            grp_col = cand
            break

    if grp_col:
        grps = df[grp_col].unique()
        np.random.shuffle(grps)
        n_train = int(SPLIT * len(grps))
        train_grps = grps[:n_train]
        val_grps = grps[n_train:]
        train_df = df[df[grp_col].isin(train_grps)].reset_index(drop=True)
        val_df = df[df[grp_col].isin(val_grps)].reset_index(drop=True)
        print(f"Group split by {grp_col}")
    else:
        idx = np.arange(len(df))
        np.random.shuffle(idx)
        n_train = int(SPLIT * len(idx))
        train_df = df.iloc[idx[:n_train]].reset_index(drop=True)
        val_df = df.iloc[idx[n_train:]].reset_index(drop=True)
        print(f"Random split: {len(train_df)}/{len(val_df)}")

    if BASELINE and BASELINE in df.columns:
        print(f"Delta learning: {TARGET} - {BASELINE}")
        train_df["_delta"] = train_df[TARGET] - train_df[BASELINE]
        val_df["_delta"] = val_df[TARGET] - val_df[BASELINE]
    else:
        print("Direct learning (no baseline)")
        train_df["_delta"] = train_df[TARGET]
        val_df["_delta"] = val_df[TARGET]

    scaler = StandardScaler()
    scaler.fit(train_df[["_delta"]].values)
    train_df["_delta_scaled"] = scaler.transform(train_df[["_delta"]].values).flatten()
    val_df["_delta_scaled"] = scaler.transform(val_df[["_delta"]].values).flatten()

    _, labels = _get_atoms(train_df)

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, f"checkpoints/delta_scaler_{TARGET}.joblib")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    func = lambda trial: objective(trial, train_df, val_df, labels)

    print(f"\nStarting {TRIALS} trials...")
    try:
        study.optimize(func, n_trials=TRIALS, show_progress_bar=True)
    except KeyboardInterrupt:
        print("Interrupted.")

    print(f"\nBest Value: {study.best_value:.6f}")
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"optuna_results_{TARGET}.csv")
    study.trials_dataframe().to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
