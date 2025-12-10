import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# -----------------------
# Config
# -----------------------
SEED = 42
DATA_PATH = "../../data/interim/urea.parquet"
TARGET = "def2-TZVP_SCF"
FEATS = [
    "STO-3G_SCF",
    "3-21G_SCF",
    "6-31G_SCF",
    "def2-SVP_SCF",
]
PLOT_PATH = "../../reports/figures/direct_lowfi_parity.png"

SPLIT = 0.75
USE_RIDGE = False
ALPHA = 1e-3

random.seed(SEED)
np.random.seed(SEED)

# -----------------------
# Load Data
# -----------------------
print(f"Loading {DATA_PATH}...")
df = pd.read_parquet(DATA_PATH)
print("Rows:", len(df), "Cols:", len(df.columns))

miss = [c for c in FEATS + [TARGET] if c not in df.columns]
if miss:
    raise ValueError(f"Missing cols: {miss}")

# -----------------------
# Split
# -----------------------
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
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    print(f"Random split: {len(train_df)}/{len(val_df)}")

# -----------------------
# Prepare & Scale
# -----------------------
X_train = train_df[FEATS].values.astype(float)
X_val = val_df[FEATS].values.astype(float)
y_train = train_df[TARGET].values.astype(float)
y_val = val_df[TARGET].values.astype(float)

train_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
val_mask = np.isfinite(X_val).all(axis=1) & np.isfinite(y_val)

if not train_mask.all() or not val_mask.all():
    print(f"Dropping {np.sum(~train_mask)} train, {np.sum(~val_mask)} val NaNs")
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_val, y_val = X_val[val_mask], y_val[val_mask]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

# -----------------------
# Train
# -----------------------
if USE_RIDGE:
    model = Ridge(alpha=ALPHA)
    print(f"Ridge (alpha={ALPHA})")
else:
    model = LinearRegression()
    print("LinearRegression")

model.fit(X_train_s, y_train)

# -----------------------
# Evaluate
# -----------------------
y_pred = model.predict(X_val_s)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print(f"Val MAE: {mae:.6f} | R2: {r2:.6f}")

coef_s = model.coef_
std = scaler.scale_
coef_orig = coef_s / std
intercept = model.intercept_ - np.dot(coef_s, scaler.mean_ / std)

print(f"Intercept: {intercept:.4f}")
print("Coefficients:")
for n, c in zip(FEATS, coef_orig):
    print(f"  {n}: {c:.6e}")

# -----------------------
# Plots
# -----------------------
Path(os.path.dirname(PLOT_PATH) or ".").mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(7, 7))
sns.scatterplot(x=y_val, y=y_pred, alpha=0.6)
mn, mx = min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())
plt.plot([mn, mx], [mn, mx], "r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Direct Regression: MAE={mae:.4f} | R2={r2:.4f}")
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(y_pred - y_val, kde=True)
plt.xlabel("Residual")
plt.title("Residuals")
plt.tight_layout()
plt.savefig(os.path.splitext(PLOT_PATH)[0] + "_residuals.png")
plt.show()
