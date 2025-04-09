import pandas as pd
import numpy as np

# Simulation settings
start_time = pd.Timestamp("12:00:00")
num_entries = 300
time_step = 5  # seconds between entries
pressures = []
labels = []

# Pressure ranges for each particle type
# These mimic the threshold regions typically observed in Cherenkov counters
ranges = {
    "proton": (900, 1000),
    "kaon": (600, 750),
    "electron": (300, 400)
}

# Simulate blocks of pressure segments
segments = [
    ("proton", 100),
    ("kaon", 100),
    ("electron", 100),
]

current_time = start_time

for label, count in segments:
    low, high = ranges[label]
    for _ in range(count):
        pressure = np.random.uniform(low, high)
        pressures.append(round(pressure, 2))
        labels.append(label)
        current_time += pd.Timedelta(seconds=time_step)

# Create DataFrame
df_sim = pd.DataFrame({
    "Timestamp": pd.date_range(start=start_time, periods=num_entries, freq=f"{time_step}s").time,
    "Pressure": pressures,
    "Particle_Label": labels
})

# Save to CSV
df_sim.to_csv("simulated_pressure_data.csv", index=False)

print("✅ Simulated pressure data saved as 'simulated_pressure_data.csv'")
