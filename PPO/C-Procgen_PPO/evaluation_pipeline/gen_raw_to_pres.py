import pandas as pd
import re
from pathlib import Path

csv_path = Path("tensorboard_logs/ailab/gen_sweep/natureCNN/results.csv")
df = pd.read_csv(csv_path)
output_dir = csv_path.parent


# Extract the level bucket from the run name
def extract_levels(run_name):
    s = str(run_name)
    m = re.search(r'levels(\d+)', s, re.IGNORECASE)
    if m:
        return m.group(1)
    raise ValueError(f"Could not extract level bucket from run name: {run_name}")


df["levels"] = df["run_name"].apply(extract_levels)

level_order = ["100", "500", "1000", "2000", "4000", "8000", "12000", "16000", "0"]
df["levels"] = pd.Categorical(df["levels"], categories=level_order, ordered=True)


# Aggregate mean/std
summary = (
    df.groupby("levels")
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

# Reset index so levels becomes a regular column and round numbers
summary = summary.reset_index()
summary = summary.round(2)
summary["levels"] = pd.Categorical(summary["levels"], categories=level_order, ordered=True)
summary = summary.sort_values("levels")

# Combine mean/std pairs into single "mean± std" strings for presentation
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

# Build new DataFrame keeping order: levels first, then combined/formatted cols, then any remaining means
out = pd.DataFrame()
out["levels"] = summary["levels"]

# add combined columns (in consistent order)
for k, series in combined.items():
    out[k] = series

# add remaining columns that were not combined (format numeric to 2 decimals)
for col in cols:
    if col in to_drop or col == "levels":
        continue
    # format floats
    if pd.api.types.is_float_dtype(summary[col]) or pd.api.types.is_integer_dtype(summary[col]):
        out[col] = summary[col].map(lambda v: f"{v:.2f}")
    else:
        out[col] = summary[col]

# Rename levels and sort rows by numeric levels when possible
out = out.rename(columns={"levels": "Levels"})
out["Levels"] = pd.Categorical(out["Levels"], categories=level_order, ordered=True)
out = out.sort_values("Levels")
summary = out

# Extract death stats into a separate table
death_stat_cols = [col for col in summary.columns if "death" in col]
if death_stat_cols:
    death_summary = summary[["Levels"] + death_stat_cols].copy()
    summary_main = summary[[col for col in summary.columns if "death" not in col]].copy()
else:
    death_summary = None
    summary_main = summary

# Save main summary Markdown
markdown_output = summary_main.to_markdown(index=False)
with open(output_dir / "levels_summary.md", "w", encoding="utf-8") as f:
    f.write(markdown_output)


# Produce a LaTeX table with a clean header and wrapped to page width
ncols = len(summary_main.columns)
column_format = "l" + "r" * (ncols - 1)
tabular = summary_main.to_latex(index=False, column_format=column_format, escape=True)

latex_output = (
    "\\begin{table}[ht]\n"
    "\\centering\n"
    "\\caption{Aggregated CoinRun level experiment results.}\n"
    "\\label{tab:level_results}\n"
    "\\resizebox{\\textwidth}{!}{%\n"
    + tabular
    + "}%\n"
    "\\end{table}\n"
)

with open(output_dir / "levels_summary.tex", "w", encoding="utf-8") as f:
    f.write(latex_output)

# Save death stats separately if present
if death_summary is not None:
    death_markdown_output = death_summary.to_markdown(index=False)
    with open(output_dir / "death_stats_summary.md", "w", encoding="utf-8") as f:
        f.write(death_markdown_output)

    # Produce a LaTeX table for death stats
    death_ncols = len(death_summary.columns)
    death_column_format = "l" + "r" * (death_ncols - 1)
    death_tabular = death_summary.to_latex(index=False, column_format=death_column_format, escape=True)

    death_latex_output = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Aggregated CoinRun level experiment death statistics.}\n"
        "\\label{tab:death_stats}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        + death_tabular
        + "}%\n"
        "\\end{table}\n"
    )

    with open(output_dir / "death_stats_summary.tex", "w", encoding="utf-8") as f:
        f.write(death_latex_output)

# Print preview and saved files
print("\n=== Main Metrics Markdown Preview ===")
print(markdown_output)

if death_summary is not None:
    print("\n=== Death Stats Markdown Preview ===")
    print(death_markdown_output)

print("\nSaved:")
print(f" - {output_dir / 'levels_summary.md'}")
print(f" - {output_dir / 'levels_summary.tex'}")
if death_summary is not None:
    print(f" - {output_dir / 'death_stats_summary.md'}")
    print(f" - {output_dir / 'death_stats_summary.tex'}")