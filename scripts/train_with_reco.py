import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Load labeled and engineered features
df_labels = pd.read_csv("bl4s_labeled_output.csv")
df_features = pd.read_csv("bl4s_engineered_features.csv")

# Merge and clean
df = pd.concat([df_features, df_labels["label"]], axis=1)
df = df.dropna()
df["label"] = df["label"].astype(str)

# Feature list
features = df_features.columns.tolist()
X = df[features]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Train
clf = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
clf.fit(X_train, y_train)
joblib.dump(clf, "bl4s_model_engineered.joblib")

# Predict
y_pred = clf.predict(X_test)

# Eval
print("\n🔍 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap="viridis")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_engineered.png")
plt.show()

# Feature importance
importances = clf.feature_importances_
sorted_idx = np.argsort(importances)
plt.figure(figsize=(8, 6))
plt.barh(np.array(features)[sorted_idx], importances[sorted_idx])
plt.xlabel("Importance")
plt.title("Feature Importance (Engineered Features)")
plt.tight_layout()
plt.savefig("feature_importance_engineered.png")
plt.show()

# Histograms by feature
for col in features:
    plt.figure()
    sns.histplot(data=df, x=col, hue="label", bins=40, kde=True, palette="Set2")
    plt.title(f"{col} by Particle Type")
    plt.savefig(f"hist_{col}.png")
    plt.close()

# Pairplot (top 4 features)
top4 = np.array(features)[sorted_idx][-4:]
sns.pairplot(df[list(top4) + ["label"]], hue="label", palette="Set2")
plt.suptitle("Pairplot of Top 4 Features", y=1.02)
plt.savefig("pairplot_top4.png")
plt.close()

# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("heatmap_corr.png")
plt.show()

print("\n✅ All done!")
print("📦 Saved:")
print(" - bl4s_model_engineered.joblib")
print(" - confusion_matrix_engineered.png")
print(" - feature_importance_engineered.png")
print(" - hist_*.png")
print(" - pairplot_top4.png")
print(" - heatmap_corr.png")
