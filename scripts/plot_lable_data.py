import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the labeled dataset
df = pd.read_csv("bl4s_labeled_output.csv")

# Drop rows with missing or invalid data
df = df.dropna(subset=["Cal0_amplitude", "timestamp", "label"])
df = df[df["label"].isin(["electron", "kaon", "proton"])]  # Sanity filter

# Create output directory for plots
os.makedirs("plots", exist_ok=True)

# Common watermark function
def add_watermark():
    plt.text(0.5, 0.9, "BL4S-CERN Work in Progress", fontsize=12, color="black",
             ha='center', transform=plt.gcf().transFigure, style='italic')

# 1️⃣ Histogram of Cal0_amplitude by Particle Type
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x="Cal0_amplitude", hue="label", bins=100, kde=True, palette="deep", element="step")
plt.title("Cal0_amplitude Distribution by Particle Type")
plt.xlabel("Cal0_amplitude")
plt.ylabel("Count")
add_watermark()
plt.tight_layout()
plt.savefig("plots/histogram_cal0_by_particle.png", dpi=300)
plt.close()

# 2️⃣ Scatter Plot: Timestamp vs Cal0_amplitude
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="timestamp", y="Cal0_amplitude", hue="label", alpha=0.6, s=10)
plt.title("Timestamp vs Cal0_amplitude (by Particle Type)")
plt.xlabel("Timestamp")
plt.ylabel("Cal0_amplitude")
plt.legend(title="Particle")
add_watermark()
plt.tight_layout()
plt.savefig("plots/timestamp_vs_cal0.png", dpi=300)
plt.close()

# 3️⃣ Count of Events per Particle Type
plt.figure(figsize=(6, 6))
sns.countplot(data=df, x="label", order=["electron", "kaon", "proton"], palette="muted")
plt.title("Number of Events per Particle Type")
plt.xlabel("Particle Type")
plt.ylabel("Count")
add_watermark()
plt.tight_layout()
plt.savefig("plots/event_count_by_label.png", dpi=300)
plt.close()

# 4️⃣ KDE plot of Cal0_amplitude for each label
plt.figure(figsize=(8, 6))
for particle in ["electron", "kaon", "proton"]:
    sns.kdeplot(data=df[df["label"] == particle], x="Cal0_amplitude", label=particle)
plt.title("KDE Plot of Cal0_amplitude by Particle Type")
plt.xlabel("Cal0_amplitude")
plt.ylabel("Density")
plt.legend(title="Particle")
add_watermark()
plt.tight_layout()
plt.savefig("plots/kde_cal0_by_particle.png", dpi=300)
plt.close()

print("✅ All plots saved in the 'plots/' directory.")
