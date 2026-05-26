import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Load CSV
df = pd.read_csv("tensorboard_logs/ailab/vis_sweep/results.csv")

# Extract visibility from context column
def extract_visibility(context_str):
    context_str = str(context_str)
    visibilities = []
    for v in [9, 11, 13, 15, 17]:
        if f'"visibility": {v}' in context_str or f'"visibility":{v}' in context_str:
            visibilities.append(v)

    if len(visibilities) == 1:
        return str(visibilities[0])
    return "Mixed"

df["visibility"] = df["context"].apply(extract_visibility)

# Ordered visibility labels
visibility_order = ["Mixed", "9", "11", "13", "15", "17"]

# Aggregate failure statistics by visibility across runs
train_cols = [
    ("train_death_saw_pct", "Saw"),
    ("train_death_enemy_pct", "Enemy"),
    ("train_death_lava_pct", "Lava"),
    ("train_death_timeout_pct", "Timeout"),
]
test_cols = [
    ("test_death_saw_pct", "Saw"),
    ("test_death_enemy_pct", "Enemy"),
    ("test_death_lava_pct", "Lava"),
    ("test_death_timeout_pct", "Timeout"),
]

train_grouped = (
    df.groupby("visibility")[[col for col, _ in train_cols]]
    .mean()
    .reindex(visibility_order)
)
test_grouped = (
    df.groupby("visibility")[[col for col, _ in test_cols]]
    .mean()
    .reindex(visibility_order)
)

max_val = max(train_grouped.max().max(), test_grouped.max().max())
y_max = max_val * 1.15 if max_val > 0 else 1

fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
axes = axes.flatten()

colors = plt.cm.Set2.colors
labels = [label for _, label in train_cols]
x_pos = np.arange(len(labels))

for idx, visibility in enumerate(visibility_order):
    ax = axes[idx]

    if visibility not in train_grouped.index or visibility not in test_grouped.index:
        ax.axis("off")
        continue

    train_values = [train_grouped.loc[visibility, col] for col, _ in train_cols]
    test_values = [test_grouped.loc[visibility, col] for col, _ in test_cols]

    width = 0.28
    offset = 0.18
    train_bars = ax.bar(
        x_pos - offset,
        train_values,
        width=width,
        color=colors[: len(labels)],
    )
    test_bars = ax.bar(
        x_pos + offset,
        test_values,
        width=width,
        color=colors[: len(labels)],
        alpha=0.6,
        hatch="///",
        edgecolor="black",
    )

    ax.set_title(f"Vis {visibility}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylim(0, y_max)

    ax.bar_label(train_bars, fmt="%.1f%%", padding=2, fontsize=8)
    ax.bar_label(test_bars, fmt="%.1f%%", padding=2, fontsize=8)

axes[0].set_ylabel("Failure Percentage (%)")
axes[3].set_ylabel("Failure Percentage (%)")

hazard_handles = [
    Patch(facecolor=colors[i], label=labels[i]) for i in range(len(labels))
]
train_handle = Patch(facecolor="white", edgecolor="black", label="Train (solid)")
test_handle = Patch(facecolor="white", edgecolor="black", hatch="///", label="Test (hatched)")

fig.legend(
    handles=hazard_handles + [train_handle, test_handle],
    loc="upper center",
    ncol=3,
    title="Hazard Colors and Split",
    bbox_to_anchor=(0.5, 0.94),
)

fig.suptitle("Failure Distribution by Visibility (Train vs Test)", y=0.985)
plt.tight_layout(rect=(0, 0, 1, 0.85))
plt.show()