
import pandas as pd
import matplotlib.pyplot as plt
import os

# ================================
# CONFIG
# ================================
files = [
    "audit_resultsfirst_Qwen_Qwen2.5-Coder-7B-Instruct.xlsx",
    "audit_resultsfirst_mistralai_Mistral-7B-Instruct-v0.3.xlsx",
    "audit_resultsfirst_deepseek-ai_deepseek-coder-6.7b-instruct.xlsx"
]

output_dir = "paper_outputs"
os.makedirs(output_dir, exist_ok=True)

scores = [
    "correct_tests_score",
    "coverage_score",
    "solution_score",
    "clarity_score"
]

# ================================
# DEGENERATE FILTER
# ================================
def is_degenerate(row):
    return (
        row["correct_tests_score"] == 1 and
        row["coverage_score"] == 1 and
        row["solution_score"] == 1 and
        row["clarity_score"] == 1
    )

# ================================
# LOAD DATA
# ================================
dfs = []
for file in files:
    try:
        df = pd.read_excel(file, engine="openpyxl")
        dfs.append(df)
    except Exception as e:
        print(f"Skipping {file}: {e}")

df = pd.concat(dfs, ignore_index=True)

# ================================
# FILTER VALID DATA
# ================================
df["degenerate"] = df.apply(is_degenerate, axis=1)
df_valid = df[~df["degenerate"]]

# ================================
# 1. GRAPH (ALL 4 CATEGORIES)
# ================================
avg_scores = df_valid.groupby("dataset")[scores].mean()

plt.figure(figsize=(8,5))
avg_scores.T.plot(kind="bar")
plt.title("Benchmark Quality Across Four Dimensions")
plt.ylabel("Score (1–5)")
plt.xticks(rotation=45)
plt.legend(title="Dataset")
plt.tight_layout()

plt.savefig(f"{output_dir}/figure_scores.png")
plt.close()

# ================================
# 2. TABLE (DETAILED SCORES)
# ================================
table_df = avg_scores.round(2)

print("\n✅ TABLE: Detailed Benchmark Scores")
print(table_df)

# ================================
# 3. SAVE TABLE AS CSV
# ================================
table_df.to_csv(f"{output_dir}/table_scores.csv")

# ================================
# 4. SAVE TABLE AS IMAGE
# ================================
fig, ax = plt.subplots(figsize=(6,2))

ax.axis('off')
table = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    rowLabels=table_df.index,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.title("Benchmark Quality Scores", pad=20)

plt.savefig(f"{output_dir}/table_scores.png", bbox_inches='tight')
plt.close()

# ================================
# 5. SUMMARY TABLE (OPTIONAL)
# ================================
summary_rows = []

for dataset in df["dataset"].unique():
    raw_avg = table_df.loc[dataset].mean()
    valid_ratio = len(df_valid[df_valid["dataset"] == dataset]) / len(df[df["dataset"] == dataset])

    summary_rows.append({
        "dataset": dataset,
        "avg_score": round(raw_avg, 2),
        "valid_ratio": round(valid_ratio, 2)
    })

summary_df = pd.DataFrame(summary_rows)

print("\n✅ TABLE: Summary")
print(summary_df)

summary_df.to_csv(f"{output_dir}/table_summary.csv")

# Save summary as image
fig, ax = plt.subplots(figsize=(5,2))
ax.axis('off')

table = ax.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.title("Summary Scores", pad=20)
plt.savefig(f"{output_dir}/table_summary.png", bbox_inches='tight')
plt.close()

print("\n✅ DONE! All outputs saved in:", output_dir)
