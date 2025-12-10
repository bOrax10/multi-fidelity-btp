import os
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# -----------------------
# Config / Dataset Paths
# -----------------------
SEED = 42
DATA_PATH = "../../data/interim/urea.parquet"
TARGET = "def2-TZVP_EV_1"
BASELINE = "def2-SVP_EV_1"
PLOT_PATH = "../../reports/figures/linear_baseline_parity.png"
SPLIT = 0.75

USE_RIDGE = False
USE_LASSO = False
ALPHA = 1e-3

random.seed(SEED)
np.random.seed(SEED)

# -----------------------
# Load and Split
# -----------------------
print(f"Loading data from {DATA_PATH}...")
df = pd.read_parquet(DATA_PATH)
print("Rows:", len(df), "Columns:", len(df.columns))

grp_col = None
for candidate in ("molecule_id", "mol_id", "traj_id", "trajectory_id"):
    if candidate in df.columns:
        grp_col = candidate
        break

if grp_col is not None:
    grps = df[grp_col].unique()
    np.random.shuffle(grps)
    n_train = int(SPLIT * len(grps))
    train_grps = grps[:n_train]
    val_grps = grps[n_train:]
    train_df = df[df[grp_col].isin(train_grps)].reset_index(drop=True)
    val_df = df[df[grp_col].isin(val_grps)].reset_index(drop=True)
    print(f"Group-split by '{grp_col}'")
else:
    idx = np.arange(len(df))
    np.random.shuffle(idx)
    n_train = int(SPLIT * len(idx))
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    print(f"Random-split: train={len(train_df)}, val={len(val_df)}")

# -----------------------
# Prepare Features and Target
# -----------------------
pat = re.compile(r"^[A-Za-z]+_\d+_[xyz]$")
coords = [col for col in df.columns if pat.match(col)]
if not coords:
    raise ValueError("No coordinate columns found.")

feats = coords.copy()
if BASELINE is not None and BASELINE in df.columns:
    feats.append(BASELINE)

print(f"Feature columns used ({len(feats)}):", feats[:6])

if BASELINE is not None and BASELINE in df.columns:
    train_df["_base"] = train_df[BASELINE].values
    val_df["_base"] = val_df[BASELINE].values
    train_df["_target"] = train_df[TARGET] - train_df["_base"]
    val_df["_target"] = val_df[TARGET] - val_df["_base"]
else:
    train_df["_base"] = 0.0
    val_df["_base"] = 0.0
    train_df["_target"] = train_df[TARGET]
    val_df["_target"] = val_df[TARGET]

# -----------------------
# Fit Linear Model
# -----------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feats].values)
X_val = scaler.transform(val_df[feats].values)
y_train = train_df["_target"].values

if USE_RIDGE:
    model = Ridge(alpha=ALPHA)
    print(f"Using Ridge(alpha={ALPHA})")
elif USE_LASSO:
    model = Lasso(alpha=ALPHA)
    print(f"Using Lasso(alpha={ALPHA})")
else:
    model = LinearRegression()
    print("Using standard LinearRegression")

model.fit(X_train, y_train)
y_pred_delta = model.predict(X_val)

y_pred = y_pred_delta + val_df["_base"].values
y_true = val_df[TARGET].values

mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f"Validation MAE: {mae:.6f} | R2: {r2:.6f}")

# -----------------------
# Plot Parity
# -----------------------
Path(os.path.dirname(PLOT_PATH) or ".").mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(7, 7))
sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
plt.plot([mn, mx], [mn, mx], "r--", label="ideal")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Linear baseline: MAE={mae:.4f} | R2={r2:.4f}")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(y_pred - y_true, kde=True)
plt.title("Residuals (pred - actual)")
plt.xlabel("Residual")
plt.tight_layout()
plt.show()
