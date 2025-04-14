import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure output directory exists
os.makedirs("visualizations", exist_ok=True)

# Load the saved AUCs
crossval_df = pd.read_csv("visualizations/crossval_auc.csv")
in_sample_df = pd.read_csv("visualizations/in_sample_auc.csv")

# --- 1. BARPLOTS for Each Set (In Sample + CrossVal) ---


def plot_auc_bars(df, title, filename):
    sets = df['Set'].unique()
    for s in sets:
        plt.figure(figsize=(10, 5))
        subset = df[df['Set'] == s].sort_values("AUC", ascending=False)
        sns.barplot(data=subset, x="Model", y="AUC", palette="viridis")
        plt.title(f"{title} - {s}")
        plt.ylim(0.45, 1.05)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"visualizations/{filename}_{s.replace(' ', '_')}.png")
        plt.close()


print("📊 Creating bar plots...")
plot_auc_bars(in_sample_df, "In-Sample AUCs", "in_sample_auc")
plot_auc_bars(crossval_df, "Cross-Validation AUCs", "crossval_auc")

# --- 2. Lineplot Comparing In-Sample vs. CrossVal AUC ---
print("📈 Creating comparison plots...")
merged = pd.merge(in_sample_df, crossval_df, on=[
                  "Set", "Model"], suffixes=("_in", "_cv"))
sets = merged["Set"].unique()

for s in sets:
    subset = merged[merged["Set"] == s]
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=subset, x="Model", y="AUC_in",
                 marker='o', label="In-Sample")
    sns.lineplot(data=subset, x="Model", y="AUC_cv",
                 marker='s', label="Cross-Validation")
    plt.title(f"In-Sample vs Cross-Validation AUCs - {s}")
    plt.ylim(0.45, 1.05)
    plt.xticks(rotation=45)
    plt.ylabel("AUC")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"visualizations/compare_auc_{s.replace(' ', '_')}.png")
    plt.close()

# --- 3. Heatmap Overview ---
print("🔥 Creating AUC heatmap overview...")
pivot_auc = merged.pivot(index="Model", columns="Set", values="AUC_cv")
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_auc, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True)
plt.title("Cross-Validation AUCs (Heatmap)")
plt.tight_layout()
plt.savefig("visualizations/crossval_auc_heatmap.png")
plt.close()

print("✅ All visualizations saved to /visualizations")
