import pandas as pd
import re
from pathlib import Path


csv_path = Path("tensorboard_logs/ailab/vis_sweep/results.csv")
df = pd.read_csv(csv_path)
output_dir = csv_path.parent


# Extract visibility robustly from the context string
def extract_visibility(context):
    s = str(context)
    m = re.search(r'"visibility"\s*:\s*(\d+)', s)
    if m:
        return m.group(1)
    return "Mixed"


df["visibility"] = df["context"].apply(extract_visibility)


# Aggregate mean/std
summary = (
    df.groupby("visibility")
    .agg({
        "train_win_rate": ["mean", "std"],
        "test_win_rate_mean": ["mean", "std"],
        "generalization_gap": ["mean", "std"],
        "train_ep_rew_mean": ["mean", "std"],
        "test_ep_rew_mean": ["mean", "std"],
        "test_death_saw_pct": "mean",
        "test_death_enemy_pct": "mean",
        "test_death_lava_pct": "mean",
    })
)

# Flatten multi-index columns
summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]

# Reset index so visibility becomes a regular column and round numbers
summary = summary.reset_index()
summary = summary.round(2)

# Combine mean/std pairs into single ``mean± std`` strings for presentation
cols = list(summary.columns)
combined = {}
to_drop = []
for col in cols:
    if col.endswith("_mean"):
        base = col[:-5]
        std_col = base + "_std"
        if std_col in cols:
            # format as string e.g. 80.33± 1.40
            combined[base] = summary[col].map(lambda v: f"{v:.2f}") + "± " + summary[std_col].map(lambda v: f"{v:.2f}")
            to_drop.extend([col, std_col])

# Build new DataFrame keeping order: visibility first, then combined/formatted cols, then any remaining means
out = pd.DataFrame()
out["visibility"] = summary["visibility"]

# add combined columns (in consistent order)
for k, series in combined.items():
    out[k] = series

# add remaining columns that were not combined (format numeric to 2 decimals)
for col in cols:
    if col in to_drop or col == "visibility":
        continue
    # format floats
    if pd.api.types.is_float_dtype(summary[col]) or pd.api.types.is_integer_dtype(summary[col]):
        out[col] = summary[col].map(lambda v: f"{v:.2f}")
    else:
        out[col] = summary[col]

# Rename visibility and sort rows by numeric visibility when possible (Mixed goes last)
out = out.rename(columns={"visibility": "Visibility"})
def _vis_key(v):
    try:
        return int(v)
    except Exception:
        return 999

out["_sort_key"] = out["Visibility"].apply(_vis_key)
out = out.sort_values("_sort_key").drop(columns=["_sort_key"]) 
summary = out


# Save Markdown
markdown_output = summary.to_markdown(index=False)
with open(output_dir / "visibility_summary.md", "w", encoding="utf-8") as f:
    f.write(markdown_output)


# Produce a LaTeX table with a clean header and wrapped to page width
ncols = len(summary.columns)
column_format = "l" + "r" * (ncols - 1)
tabular = summary.to_latex(index=False, column_format=column_format, escape=True)

latex_output = (
    "\\begin{table}[ht]\n"
    "\\centering\n"
    "\\caption{Aggregated CoinRun visibility experiment results.}\n"
    "\\label{tab:visibility_results}\n"
    "\\resizebox{\\textwidth}{!}{%\n"
    + tabular
    + "}%\n"
    "\\end{table}\n"
)

with open(output_dir / "visibility_summary.tex", "w", encoding="utf-8") as f:
    f.write(latex_output)


# Print preview and saved files
print("\n=== Markdown Preview ===\n")
print(markdown_output)
print("\nSaved:")
print(f" - {output_dir / 'visibility_summary.md'}")
print(f" - {output_dir / 'visibility_summary.tex'}")