import uproot
import pandas as pd
import numpy as np
from tqdm import tqdm

# File paths
root_file = "1695788924.root"
pressure_csv = "simulated_pressure_data.csv"
output_csv = "bl4s_labeled_output.csv"
output_root = "bl4s_labeled_output.root"

# Load simulated pressure+label file
df_pressure = pd.read_csv(pressure_csv)

# Convert timestamp to seconds past midnight
df_pressure["Timestamp_sec"] = pd.to_datetime(df_pressure["Timestamp"], format="%H:%M:%S").dt.hour * 3600 + \
                               pd.to_datetime(df_pressure["Timestamp"], format="%H:%M:%S").dt.minute * 60 + \
                               pd.to_datetime(df_pressure["Timestamp"], format="%H:%M:%S").dt.second

# Open ROOT file and access TTree
with uproot.open(root_file) as file:
    reco_tree = file["RECOdata"]
    
    # Load relevant branches
    cal_amp = reco_tree.arrays(["Cal0_amplitude"], library="pd")
    n_entries = len(cal_amp)

    # Simulate timestamps over pressure range
    start_sec = df_pressure["Timestamp_sec"].min()
    end_sec = df_pressure["Timestamp_sec"].max()
    sim_timestamps = np.linspace(start_sec, end_sec, n_entries)

    # Map timestamps to particle labels
    pressure_times = df_pressure["Timestamp_sec"].values
    pressure_labels = df_pressure["Particle_Label"].values

    def get_label(ts):
        idx = np.searchsorted(pressure_times, ts, side='right') - 1
        return pressure_labels[idx] if 0 <= idx < len(pressure_labels) else "unknown"

    labels = [get_label(t) for t in sim_timestamps]

    # Combine into labeled DataFrame
    df_labeled = cal_amp.copy()
    df_labeled["timestamp"] = sim_timestamps
    df_labeled["label"] = labels

    # Save as CSV
    df_labeled.to_csv(output_csv, index=False)

    # Save as ROOT file
    with uproot.recreate(output_root) as f:
        f["LabeledData"] = df_labeled

print("✅ Labeled dataset saved:")
print(f"- CSV: {output_csv}\n- ROOT: {output_root}")
