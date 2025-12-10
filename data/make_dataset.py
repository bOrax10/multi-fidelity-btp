import pandas as pd
import numpy as np
from glob import glob
from collections import Counter

files = glob("../../data/raw/*.npz")

QeMFi_df = pd.DataFrame()

atomic_number_to_symbol = {1: "H", 6: "C", 7: "N", 8: "O"}

fidelities = {1: "STO-3G", 2: "3-21G", 3: "6-31G", 4: "def2-SVP", 5: "def2-TZVP"}


def convert_npz_to_df(data):
    """
    Converts molecular data (R, SCF, and EV) from a .npz-like structure
    to a single, combined pandas DataFrame.
    """

    # === Part 1: Cartesian Coordinates (R) ===
    counts = Counter()
    atom_labels = []
    for z in data["Z"]:
        symbol = atomic_number_to_symbol[z]
        counts[symbol] += 1
        atom_labels.append(f"{symbol}_{counts[symbol]}")

    columns_R = [f"{label}_{dim}" for label in atom_labels for dim in ["x", "y", "z"]]
    df_coords = pd.DataFrame(
        data["R"].reshape(data["R"].shape[0], -1), columns=columns_R
    )

    # === Part 2: Ground State Energies (SCF) ===
    SCF = data["SCF"]
    columns_SCF = [f"{fidelities[i + 1]}_SCF" for i in range(SCF.shape[1])]
    df_scf = pd.DataFrame(SCF, columns=columns_SCF)

    # === Part 3: Excitation Energies (EV) ===
    EV = data["EV"]
    reshaped_EV = EV.reshape(EV.shape[0], -1)

    num_fidelities = EV.shape[1]
    num_energies = EV.shape[2]

    columns_EV = [
        f"{fidelities[i + 1]}_EV_{j + 1}"
        for i in range(num_fidelities)
        for j in range(num_energies)
    ]

    df_ev = pd.DataFrame(reshaped_EV, columns=columns_EV)

    # === Part 4: Transition Dipole Moments (TrDP) ===
    TrDP = data["TrDP"]
    num_fidelities = TrDP.shape[1]
    num_dipole_moments = TrDP.shape[2]

    columns_TrDP = [
        f"{fidelities[i + 1]}_TrDP_{j + 1}_{dir}"
        for i in range(num_fidelities)
        for j in range(num_dipole_moments)
        for dir in ["x", "y", "z"]
    ]

    reshaped_TrDP = TrDP.reshape(TrDP.shape[0], -1)
    df_trdp = pd.DataFrame(reshaped_TrDP, columns=columns_TrDP)

    # === Part 5: Oscillator Frequency (fosc) ===
    fosc = data["fosc"]
    num_fidelities = fosc.shape[1]
    num_fosc = fosc.shape[2]
    columns_fosc = [
        f"{fidelities[i + 1]}_fosc_{j + 1}"
        for i in range(num_fidelities)
        for j in range(num_fosc)
    ]

    reshaped_fosc = fosc.reshape(fosc.shape[0], -1)
    df_fosc = pd.DataFrame(reshaped_fosc, columns=columns_fosc)

    # === Part 6: Electronic contribution of Dipole Moment (DPe) ===
    DPe = data["DPe"]
    num_fidelities = DPe.shape[1]
    columns_DPe = [
        f"{fidelities[i + 1]}_DPe_{dir}"
        for i in range(num_fidelities)
        for dir in ["x", "y", "z"]
    ]

    reshaped_DPe = DPe.reshape(DPe.shape[0], -1)
    df_DPe = pd.DataFrame(reshaped_DPe, columns=columns_DPe)

    # === Part 7: Nuclear contribution of Dipole Moment (DPe) ===
    DPn = data["DPn"]
    num_fidelities = DPn.shape[1]
    columns_DPn = [
        f"{fidelities[i + 1]}_DPn_{dir}"
        for i in range(num_fidelities)
        for dir in ["x", "y", "z"]
    ]

    reshaped_DPn = DPn.reshape(DPn.shape[0], -1)
    df_DPn = pd.DataFrame(reshaped_DPn, columns=columns_DPn)

    # === Part 8: Rotational Constants (RCo) ===
    RCo = data["RCo"]
    num_fidelities = RCo.shape[1]
    columns_RCo = [
        f"{fidelities[i + 1]}_RCo_{dir}"
        for i in range(num_fidelities)
        for dir in ["a", "b", "c"]
    ]

    reshaped_RCo = RCo.reshape(RCo.shape[0], -1)
    df_RCo = pd.DataFrame(reshaped_RCo, columns=columns_RCo)

    # === Part 9: Dipole Moment along Rotational axes (DPRo) ===
    DPRo = data["DPRo"]
    num_fidelities = DPRo.shape[1]
    columns_DPRo = [
        f"{fidelities[i + 1]}_DPRo_{dir}"
        for i in range(num_fidelities)
        for dir in ["a", "b", "c"]
    ]

    reshaped_DPRo = DPRo.reshape(DPRo.shape[0], -1)
    df_DPRo = pd.DataFrame(reshaped_DPRo, columns=columns_DPRo)

    # === Part 10: Combine all DataFrames ===
    df_final = pd.concat(
        [df_coords, df_scf, df_ev, df_trdp, df_fosc, df_DPe, df_DPn, df_RCo, df_DPRo],
        axis=1,
    )

    return df_final


dataframes = {}

for file in files:
    molecule = file.split("_")[-1][:-4]
    data = np.load(file)
    file_df = convert_npz_to_df(data)
    dataframes[molecule] = file_df
    file_df["molecule"] = molecule
    QeMFi_df = pd.concat([QeMFi_df, file_df], ignore_index=True)

acrolein_df = dataframes["acrolein"].drop(columns=["molecule"])
alanine_df = dataframes["alanine"].drop(columns=["molecule"])
dmabn_df = dataframes["dmabn"].drop(columns=["molecule"])
nitrophenol_df = dataframes["nitrophenol"].drop(columns=["molecule"])
ortho_hbdi_df = dataframes["o-hbdi"].drop(columns=["molecule"])
sma_df = dataframes["sma"].drop(columns=["molecule"])
thymine_df = dataframes["thymine"].drop(columns=["molecule"])
urea_df = dataframes["urea"].drop(columns=["molecule"])
urocanic_df = dataframes["urocanic"].drop(columns=["molecule"])

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

QeMFi_df.to_parquet("../../data/interim/QeMFi.parquet", engine="pyarrow", index=False)

for file_name in dataframes.keys():
    df = dataframes[file_name].drop(columns=["molecule"])
    df.to_parquet(
        f"../../data/interim/{file_name}.parquet",
        engine="pyarrow",
        index=False,
    )
