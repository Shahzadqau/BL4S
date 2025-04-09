import uproot
import pandas as pd
import numpy as np

# --- File paths ---
root_file = "1695788924.root"
output_csv = "bl4s_engineered_features.csv"

# --- Load ROOT file and RECOdata ---
with uproot.open(root_file) as file:
    tree = file["RECOdata"]
    
    branches = [
        "NDWC0_xPosition", "NDWC1_xPosition", "NDWC2_xPosition", "NDWC3_xPosition",
        "NDWC0_yPosition", "NDWC1_yPosition", "NDWC2_yPosition", "NDWC3_yPosition",
        "NS0_time", "NS1_time", "NS2_time", "NS3_time", "NS4_time",
        "NC0_time", "NC1_time",
        "Cal0_amplitude", "Cal2_amplitude", "Cal4_amplitude", "Cal7_amplitude", "Cal8_amplitude"
    ]
    
    df = tree.arrays(branches, library="pd")

# --- Compute derived features ---
# DWC track trajectory (slope proxies)
df["dx_01"] = df["NDWC1_xPosition"] - df["NDWC0_xPosition"]
df["dx_12"] = df["NDWC2_xPosition"] - df["NDWC1_xPosition"]
df["dx_23"] = df["NDWC3_xPosition"] - df["NDWC2_xPosition"]

df["dy_01"] = df["NDWC1_yPosition"] - df["NDWC0_yPosition"]
df["dy_12"] = df["NDWC2_yPosition"] - df["NDWC1_yPosition"]
df["dy_23"] = df["NDWC3_yPosition"] - df["NDWC2_yPosition"]

# Time-of-flight (between scintillator hits)
df["tof_01"] = df["NS1_time"] - df["NS0_time"]
df["tof_12"] = df["NS2_time"] - df["NS1_time"]
df["tof_23"] = df["NS3_time"] - df["NS2_time"]

# Calorimeter sum amplitude (energy deposit proxy)
df["calo_sum"] = (
    df["Cal0_amplitude"] +
    df["Cal2_amplitude"] +
    df["Cal4_amplitude"] +
    df["Cal7_amplitude"] +
    df["Cal8_amplitude"]
)

# --- Clean up ---
features = [
    "dx_01", "dx_12", "dx_23",
    "dy_01", "dy_12", "dy_23",
    "tof_01", "tof_12", "tof_23",
    "calo_sum"
]
df_clean = df[features].dropna().reset_index(drop=True)

# --- Save output ---
df_clean.to_csv(output_csv, index=False)
print(f"✅ Feature dataset saved as: {output_csv}")
