import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df_features = pd.read_csv("bl4s_engineered_features.csv")
df_labels = pd.read_csv("bl4s_labeled_output.csv")

# Merge based on row index (assumes same order)
df = pd.concat([df_features, df_labels["label"]], axis=1)
df = df.dropna()
df = df[df["label"].isin(["electron", "kaon", "proton"])]  # Sanity filter

# Create output directory
plot_dir = "plots_engineered_labeled"
os.makedirs(plot_dir, exist_ok=True)

# Watermark function
def add_watermark():
    plt.text(0.5, 0.9, "BL4S-CERN Work in Progress", fontsize=12,
             color="black", ha='center', transform=plt.gcf().transFigure, style='italic')

# 1️⃣ Feature-wise Histograms
for col in df_features.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=col, hue="label", bins=60, kde=True, palette="Set2", element="step", common_norm=False)
    plt.title(f"{col} Distribution by Particle Type")
    plt.xlabel(col)
    plt.ylabel("Count")
    add_watermark()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"hist_{col}_by_label.png"), dpi=300)
    plt.close()

# 2️⃣ Pairplot (top features)
top_features = ["calo_sum", "tof_01", "dx_01", "dy_01"]
sns.pairplot(df[top_features + ["label"]], hue="label", palette="Set2", diag_kind="kde", plot_kws={"alpha": 0.6, "s": 20})
plt.suptitle("Pairplot of Top Engineered Features", y=1.02)
plt.savefig(os.path.join(plot_dir, "pairplot_top_features.png"), dpi=300)
plt.close()

# 3️⃣ 2D Scatter: TOF vs. Calo Sum
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="tof_01", y="calo_sum", hue="label", palette="Set2", alpha=0.7)
plt.title("TOF vs. Calorimeter Energy Sum")
plt.xlabel("TOF_01")
plt.ylabel("Calo Sum")
add_watermark()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "scatter_tof_vs_calo.png"), dpi=300)
plt.close()

# 4️⃣ 2D Scatter: dx vs. dy
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="dx_01", y="dy_01", hue="label", palette="Set2", alpha=0.7)
plt.title("Track Slope: dx_01 vs dy_01")
plt.xlabel("dx_01")
plt.ylabel("dy_01")
add_watermark()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "scatter_dx_vs_dy.png"), dpi=300)
plt.close()

print("✅ All plots saved in 'plots_engineered_labeled/'")
