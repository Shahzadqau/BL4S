import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the engineered features
df = pd.read_csv("bl4s_engineered_features.csv")

# Create output directory
plot_dir = "plots_engineered"
os.makedirs(plot_dir, exist_ok=True)

# Plot histograms for each feature
for col in df.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=col, bins=50, kde=True, color="steelblue", edgecolor="black")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.text(0.5, 0.92, "BL4S-CERN Work in Progress", fontsize=12,
             color="gray", ha='center', transform=plt.gcf().transFigure, style='italic')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"hist_{col}.png"), dpi=300)
    plt.close()
