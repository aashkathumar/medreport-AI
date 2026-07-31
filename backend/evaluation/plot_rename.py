import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

csv_path = Path(__file__).parent / "comparison_results.csv"
if not csv_path.exists():
    print("Run python evaluation/compare_providers.py first!")
    exit()

df = pd.read_csv(csv_path)

# Filter out rows that errored out
df_valid = df[df["error"].isna() | (df["error"] == "")].copy()

# Print text summary table
print("\n=== SUMMARY TABLE ===")
summary = df_valid.groupby("provider").agg(
    Avg_Response_Time_Sec=("response_time_sec", "mean"),
    Avg_Readability_Measures=("fk_grade_what_it_measures", "mean"),
    Avg_Readability_Means=("fk_grade_what_result_means", "mean")
).reset_index()
print(summary.to_string(index=False))

# Plot Charts
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Chart 1: Latency
sns.barplot(data=df_valid, x="provider", y="response_time_sec", ax=axes[0], palette="Blues_d")
axes[0].set_title("Average Latency (Lower is Better)")
axes[0].set_ylabel("Seconds")

# Chart 2: Readability
sns.barplot(data=df_valid, x="provider", y="fk_grade_what_result_means", ax=axes[1], palette="Greens_d")
axes[1].set_title("Flesch-Kincaid Readability (Grade Level)")
axes[1].set_ylabel("Grade Level (Target: ~8)")

plt.tight_layout()
output_chart = Path(__file__).parent / "evaluation_metrics.png"
plt.savefig(output_chart, dpi=300)
print(f"\nChart saved successfully to {output_chart}")