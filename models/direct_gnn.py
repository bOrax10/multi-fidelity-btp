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
from torch.nn import Linear, Dropout, ReLU
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import SchNet
from torch_geometric.nn import global_add_pool, radius_graph
from tqdm import tqdm

# -----------------------
# Config / Hyperparams
# -----------------------
SEED = 42
DATA_PATH = "../../data/interim/urea.parquet"
TARGET = "def2-TZVP_TrDP_1_x"
PLOT_PATH = f"../../reports/figures/gnn_actual_vs_predicted_{TARGET}.png"

BATCH_SIZE = 16
EPOCHS = 200
LR = 1.5469e-5
WEIGHT_DECAY = 1e-5
CUTOFF = 6.5672
PATIENCE = 30
GRAD_CLIP = 5.0
SPLIT = 0.8

HIDDEN = 128
FILTERS = 128
INTERACTIONS = 4
GAUSSIANS = 100
DROPOUT = 0.4565

LF_COLS = [
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


def make_graphs(df: pd.DataFrame, target_col: str, labels=None, extra=None):
    if labels is None:
        _, labels = _get_atoms(df)
    z_tensor, _ = _get_atoms(df)

    cols_flat = [f"{l}_{d}" for l in labels for d in ["x", "y", "z"]]
    coords = df[cols_flat].values
    n_frames = len(df)
    n_atoms = len(labels)
    pos = torch.tensor(coords.reshape(n_frames, n_atoms, 3), dtype=torch.float)
    y = torch.tensor(df[target_col].values, dtype=torch.float)

    if extra is not None:
        extra_t = torch.tensor(extra, dtype=torch.float)
    else:
        extra_t = None

    graphs = []
    for i in range(n_frames):
        kwargs = dict(
            x=z_tensor.clone(),
            pos=pos[i].clone(),
            y=y[i].clone().unsqueeze(0),
        )
        if extra_t is not None:
            kwargs["extra"] = extra_t[i].clone().unsqueeze(0)
        graphs.append(Data(**kwargs))
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
        n_extra: int = 0,
        max_num_neighbors: int = 32,
    ):
        super(GNNModel, self).__init__()
        self.max_num_neighbors = max_num_neighbors
        self.cutoff = cutoff
        self.schnet = SchNet(
            hidden_channels=hidden,
            num_filters=filters,
            num_interactions=interactions,
            num_gaussians=GAUSSIANS,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
        )
        self.n_extra = n_extra
        in_dim = hidden + (n_extra if n_extra > 0 else 0)
        self.head = torch.nn.Sequential(
            Linear(in_dim, hidden // 2),
            ReLU(),
            Dropout(0.1),
            Linear(hidden // 2, 1),
        )

    def forward(self, data):
        z = data.x.long()
        pos = data.pos
        batch = data.batch

        h = self.schnet.embedding(z)

        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            max_num_neighbors=self.max_num_neighbors,
        )
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.schnet.distance_expansion(edge_weight)

        for interaction in self.schnet.interactions:
            h = interaction(h, edge_index, edge_weight, edge_attr)

        out = global_add_pool(h, batch)

        if out.dim() == 1:
            out = out.unsqueeze(0)

        if self.n_extra > 0 and hasattr(data, "extra") and data.extra is not None:
            extra = data.extra
            if extra.dim() == 1:
                extra = extra.unsqueeze(0)
            out = torch.cat([out, extra.to(out.device).to(out.dtype)], dim=1)

        res = self.head(out)
        return res.squeeze(1)


# -----------------------
# Training Routine
# -----------------------
def train_gnn(
    data_path: str,
    target: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    cutoff: float = CUTOFF,
    plot_path: str = PLOT_PATH,
):
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print("Rows:", len(df), "Columns:", df.columns.tolist())

    grp_col = None
    for cand in ("molecule_id", "mol_id", "traj_id", "trajectory_id"):
        if cand in df.columns:
            grp_col = cand
            break

    if grp_col is not None:
        grps = df[grp_col].unique()
        np.random.shuffle(grps)
        n_train = int(SPLIT * len(grps))
        train_grps = grps[:n_train]
        val_grps = grps[n_train:]
        train_df = df[df[grp_col].isin(train_grps)].reset_index(drop=True)
        val_df = df[df[grp_col].isin(val_grps)].reset_index(drop=True)
    else:
        idx = np.arange(len(df))
        np.random.shuffle(idx)
        n_train = int(SPLIT * len(idx))
        train_idx, val_idx = idx[:n_train], idx[n_train:]
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

    y_scaler = StandardScaler()
    y_train_raw = train_df[[target]].values
    y_scaler.fit(y_train_raw)
    train_df["_y_scaled"] = y_scaler.transform(train_df[[target]].values).flatten()
    val_df["_y_scaled"] = y_scaler.transform(val_df[[target]].values).flatten()

    lf_avail = [c for c in LF_COLS if c in df.columns]
    n_extra = len(lf_avail)

    if n_extra > 0:
        lf_scaler = StandardScaler()
        lf_scaler.fit(train_df[lf_avail].values)
        train_extra = lf_scaler.transform(train_df[lf_avail].values)
        val_extra = lf_scaler.transform(val_df[lf_avail].values)
    else:
        lf_scaler = None
        train_extra = None
        val_extra = None

    _, labels = _get_atoms(train_df)
    train_ds = make_graphs(train_df, "_y_scaled", labels=labels, extra=train_extra)
    val_ds = make_graphs(val_df, "_y_scaled", labels=labels, extra=val_extra)

    print(f"Dataset sizes — train: {len(train_ds)}, val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = GNNModel(
        hidden=HIDDEN,
        filters=FILTERS,
        interactions=INTERACTIONS,
        cutoff=cutoff,
        n_extra=n_extra,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    loss_fn = torch.nn.SmoothL1Loss()

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_model_{target}.pt"

    best_val = float("inf")
    patience_cnt = 0
    best_epoch = -1
    train_hist, val_hist = [], []

    print(f"Starting training for: {target}")
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
                    "lf_cols": lf_avail,
                    "lf_scaler": lf_scaler,
                    "y_scale": y_scaler.scale_,
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

    preds_orig = y_scaler.inverse_transform(all_preds).flatten()
    targets_orig = y_scaler.inverse_transform(all_targets).flatten()

    mae = mean_absolute_error(targets_orig, preds_orig)
    r2 = r2_score(targets_orig, preds_orig)
    print(f"Final Validation MAE: {mae:.6f} | R2: {r2:.6f}")

    Path(os.path.dirname(plot_path) or ".").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_hist, label="Train")
    plt.plot(val_hist, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (scaled)")
    plt.yscale("log")
    plt.legend()
    plt.title("Learning Curves")
    plt.tight_layout()
    plt.savefig(os.path.splitext(plot_path)[0] + "_learning_curves.png")
    plt.show()

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=targets_orig, y=preds_orig, alpha=0.6)
    mn, mx = min(targets_orig.min(), preds_orig.min()), max(
        targets_orig.max(), preds_orig.max()
    )
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Parity: MAE={mae:.4f} | R2={r2:.4f}")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

    res = preds_orig - targets_orig
    plt.figure(figsize=(8, 5))
    sns.histplot(res, kde=True)
    plt.xlabel("Residual (Pred - Actual)")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(os.path.splitext(plot_path)[0] + "_residuals.png")
    plt.show()


if __name__ == "__main__":
    train_gnn(
        DATA_PATH,
        TARGET,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        cutoff=CUTOFF,
        plot_path=PLOT_PATH,
    )
