import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# Load CSV
csv_path = "tensorboard_logs/ailab/gen_sweep/natureCNN/results.csv"
output_dir = os.path.dirname(csv_path)
df = pd.read_csv(csv_path)

# Extract the level bucket from the run name.
def extract_levels(run_name):
    run_name = str(run_name)
    match = re.search(r"levels(\d+)", run_name, re.IGNORECASE)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract level bucket from run name: {run_name}")


df["levels"] = df["run_name"].apply(extract_levels)

# Aggregate statistics (train stats across runs, test stats across seeds within runs)

df["test_win_rate_seed_mean"] = df[[
    "test_win_rate_1",
    "test_win_rate_2",
    "test_win_rate_3",
]].mean(axis=1)

df["test_win_rate_seed_std"] = df[[
    "test_win_rate_1",
    "test_win_rate_2",
    "test_win_rate_3",
]].std(axis=1, ddof=0)

has_test_ep_rew_mean = "test_ep_rew_mean" in df.columns
has_test_ep_rew_seeds = all(
    col in df.columns
    for col in ("test_ep_rew_mean_1", "test_ep_rew_mean_2", "test_ep_rew_mean_3")
)

if has_test_ep_rew_seeds:
    df["test_ep_rew_mean_seed_mean"] = df[[
        "test_ep_rew_mean_1",
        "test_ep_rew_mean_2",
        "test_ep_rew_mean_3",
    ]].mean(axis=1)
    df["test_ep_rew_mean_seed_std"] = df[[
        "test_ep_rew_mean_1",
        "test_ep_rew_mean_2",
        "test_ep_rew_mean_3",
    ]].std(axis=1, ddof=0)

grouped = df.groupby("levels").agg(
    train_win_rate_mean=("train_win_rate", "mean"),
    train_win_rate_std=("train_win_rate", "std"),
    test_win_rate_mean=("test_win_rate_seed_mean", "mean"),
    test_win_rate_std=("test_win_rate_seed_mean", "std"),
    train_ep_rew_mean_mean=("train_ep_rew_mean", "mean"),
    train_ep_rew_mean_std=("train_ep_rew_mean", "std"),
)

if has_test_ep_rew_seeds:
    grouped["test_ep_rew_mean_mean"] = (
        df.groupby("levels")["test_ep_rew_mean_seed_mean"].mean()
    )
    grouped["test_ep_rew_mean_std"] = (
        df.groupby("levels")["test_ep_rew_mean_seed_mean"].std()
    )
elif has_test_ep_rew_mean:
    grouped["test_ep_rew_mean_mean"] = df.groupby("levels")["test_ep_rew_mean"].mean()
    grouped["test_ep_rew_mean_std"] = df.groupby("levels")["test_ep_rew_mean"].std()

# Ensure correct order
level_order = ["100", "500", "1000", "2000", "4000", "8000", "12000", "16000", "0"]
grouped = grouped.reindex(level_order)

# Plot 1: Levels vs Train/Test Win Rate
x = np.arange(len(grouped.index))

plt.figure(figsize=(6, 6))

train_mean = grouped["train_win_rate_mean"].to_numpy()
train_std = grouped["train_win_rate_std"].to_numpy()
test_mean = grouped["test_win_rate_mean"].to_numpy()
test_std = grouped["test_win_rate_std"].to_numpy()

plt.plot(x, train_mean, marker="o", label="Train Win Rate")
plt.fill_between(x, train_mean - train_std, train_mean + train_std, alpha=0.2)

plt.plot(x, test_mean, marker="o", label="Test Win Rate")
plt.fill_between(x, test_mean - test_std, test_mean + test_std, alpha=0.2)

plt.xticks(x, grouped.index)
plt.xlabel("Levels")
plt.ylabel("Win Rate (%)")
plt.title("Train/Test Win Rate")
plt.legend(loc="upper right")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "levels_vs_winrate.png"), dpi=300)


# Plot 2: Levels vs Training Episode Reward
plt.figure(figsize=(6, 6))

train_rew_mean = grouped["train_ep_rew_mean_mean"].to_numpy()
train_rew_std = grouped["train_ep_rew_mean_std"].to_numpy()

plt.plot(x, train_rew_mean, marker="o", label="Train Episode Reward")
plt.fill_between(x, train_rew_mean - train_rew_std, train_rew_mean + train_rew_std, alpha=0.2)

if "test_ep_rew_mean_mean" in grouped.columns:
    test_rew_mean = grouped["test_ep_rew_mean_mean"].to_numpy()
    test_rew_std = grouped["test_ep_rew_mean_std"].to_numpy()
    plt.plot(x, test_rew_mean, marker="o", label="Test Episode Reward")
    plt.fill_between(x, test_rew_mean - test_rew_std, test_rew_mean + test_rew_std, alpha=0.2)

plt.xticks(x, grouped.index)
plt.xlabel("Levels")
plt.ylabel("Mean Episode Reward")
plt.title("Train/Test Episode Reward")
plt.legend(loc="upper right")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir,"levels_vs_reward.png"), dpi=300)


plt.show()
print("\nSaved figures:")
print(" - levels_vs_winrate.png")
print(" - levels_vs_reward.png")