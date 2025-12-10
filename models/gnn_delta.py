import os
import random
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.nn import Linear, Dropout, ReLU, SiLU
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import SchNet
from tqdm import tqdm


# -----------------------
# Config / Hyperparams
# -----------------------
SEED = 42
DATA_PATH = "../../data/interim/urea.parquet"
TARGET = "def2-TZVP_TrDP_1_x"
BASE = "def2-SVP_TrDP_1_x"
PLOT_PATH = f"../../reports/figures/gnn_actual_vs_predicted_delta_{TARGET}.png"

BATCH_SIZE = 64
EPOCHS = 200
LR = 0.00077
WEIGHT_DECAY = 8.9709e-6
CUTOFF = 7
PATIENCE = 50
GRAD_CLIP = 5.0
SPLIT = 0.75

HIDDEN = 256
FILTERS = 256
INTERACTIONS = 3
GAUSSIANS = 50
DROPOUT = 0.1699


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
        hidden=HIDDEN,
        filters=FILTERS,
        interactions=INTERACTIONS,
        cutoff=CUTOFF,
    ):
        super(GNNModel, self).__init__()
        self.schnet = SchNet(
            hidden_channels=hidden,
            num_filters=filters,
            num_interactions=interactions,
            num_gaussians=GAUSSIANS,
            cutoff=cutoff,
        )
        self.head = torch.nn.Sequential(
            Linear(hidden, hidden // 2),
            SiLU(),
            Dropout(DROPOUT),
            Linear(hidden // 2, 1),
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
# Training Routine
# -----------------------
def train_delta(
    data_path: str,
    target: str,
    base: str = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    cutoff: float = CUTOFF,
    plot_path: str = PLOT_PATH,
):
    print(f"Loading {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Rows: {len(df)}, Cols: {len(df.columns)}")

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

    if base and base in df.columns:
        print(f"Baseline: {base}")
        train_df["_base"] = train_df[base].values
        val_df["_base"] = val_df[base].values
    else:
        print("No baseline found. Learning target directly.")
        train_df["_base"] = 0.0
        val_df["_base"] = 0.0

    train_df["_delta"] = train_df[target] - train_df["_base"]
    val_df["_delta"] = val_df[target] - val_df["_base"]

    scaler = StandardScaler()
    scaler.fit(train_df[["_delta"]].values)
    train_df["_delta_s"] = scaler.transform(train_df[["_delta"]].values).flatten()
    val_df["_delta_s"] = scaler.transform(val_df[["_delta"]].values).flatten()

    _, labels = _get_atoms(train_df)
    train_ds = make_graphs(train_df, "_delta_s", labels=labels)
    val_ds = make_graphs(val_df, "_delta_s", labels=labels)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = GNNModel(
        hidden=HIDDEN,
        filters=FILTERS,
        interactions=INTERACTIONS,
        cutoff=cutoff,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    loss_fn = torch.nn.SmoothL1Loss()

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_model_delta_{target}.pt"

    best_val = float("inf")
    patience_cnt = 0
    best_epoch = -1
    train_hist, val_hist = [], []

    print(f"Training for {target}...")
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_graphs = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Ep {epoch} [Train]", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            tgt = batch.y.view(-1).to(device)
            loss = loss_fn(out, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            t_loss += loss.item() * batch.num_graphs
            t_graphs += batch.num_graphs

        avg_t = t_loss / max(1, t_graphs)
        train_hist.append(avg_t)

        model.eval()
        v_loss, v_graphs = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch} [Val]", leave=False):
                batch = batch.to(device)
                out = model(batch)
                tgt = batch.y.view(-1).to(device)
                loss = loss_fn(out, tgt)
                v_loss += loss.item() * batch.num_graphs
                v_graphs += batch.num_graphs

        avg_v = v_loss / max(1, v_graphs)
        val_hist.append(avg_v)
        scheduler.step(avg_v)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Ep {epoch:03d} | Train: {avg_t:.6f} | Val: {avg_v:.6f}")

        if avg_v < best_val - 1e-8:
            best_val = avg_v
            patience_cnt = 0
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": float(best_val),
                    "delta_scaler": scaler,
                    "base_col": base,
                },
                str(ckpt_path),
            )
            print(f">>> Val improved to {best_val:.6f}. Saved {ckpt_path}")
        else:
            patience_cnt += 1
            print(f"    No improvement. Patience: {patience_cnt}/{PATIENCE}")

        if patience_cnt >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        scaler = ckpt.get("delta_scaler", scaler)
        print(
            f"Loaded best model (Epoch {ckpt.get('epoch')}, Val {ckpt.get('val_loss')})"
        )
    else:
        print("No checkpoint found; using final weights.")

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final eval"):
            batch = batch.to(device)
            out = model(batch)
            all_preds.append(out.cpu())
            all_targets.append(batch.y.view(-1).cpu())

    all_preds = torch.cat(all_preds).numpy().reshape(-1, 1)
    all_targets = torch.cat(all_targets).numpy().reshape(-1, 1)

    preds_delta = scaler.inverse_transform(all_preds).flatten()
    targets_delta = scaler.inverse_transform(all_targets).flatten()

    val_base = val_df["_base"].values
    preds_final = preds_delta + val_base
    targets_final = targets_delta + val_base

    mae = mean_absolute_error(targets_final, preds_final)
    r2 = r2_score(targets_final, preds_final)
    print(f"Final Validation MAE: {mae:.6f} | R2: {r2:.6f}")

    Path(os.path.dirname(plot_path) or ".").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_hist, label="Train")
    plt.plot(val_hist, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (scaled)")
    plt.yscale("log")
    plt.legend()
    plt.title("Learning Curves (Delta)")
    plt.tight_layout()
    plt.savefig(os.path.splitext(plot_path)[0] + "_learning_curves.png")
    plt.show()

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=targets_final, y=preds_final, alpha=0.6)
    mn, mx = min(targets_final.min(), preds_final.min()), max(
        targets_final.max(), preds_final.max()
    )
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Parity: MAE={mae:.4f} | R2={r2:.4f}")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

    res = preds_final - targets_final
    plt.figure(figsize=(8, 5))
    sns.histplot(res, kde=True)
    plt.xlabel("Residual")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(os.path.splitext(plot_path)[0] + "_residuals.png")
    plt.show()


if __name__ == "__main__":
    train_delta(
        DATA_PATH,
        TARGET,
        base=BASE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        cutoff=CUTOFF,
        plot_path=PLOT_PATH,
    )
