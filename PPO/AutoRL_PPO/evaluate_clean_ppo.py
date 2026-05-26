import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFERRED_VENV_PYTHON = Path(
    r"C:\Users\danim\PycharmProjects\GroupProject\.venv\Scripts\python.exe"
)
REEXEC_GUARD_ENV = "EVALUATE_CLEAN_PPO_REEXEC_DONE"
WIN_RATE_BLOCK_EPISODES = 100
ORDERED_ENVS = ("coinrun", "starpilot", "climber", "jumper", "ninja", "heist")
DEFAULT_EVAL_SEEDS = (1, 10, 2, 3)
EVALUATION_PROTOCOL = "eval_seeds_1_2_3_10"
INDIVIDUAL_REFERENCE_SEED = 10
DEFAULT_SEED_TIMEOUT_SECONDS = 7 * 60
MANUAL_SKIPPED_RUNS = {
    "clean_ppo_m_heist_50M_H_834710": (
        "manual skip: hard Heist Clean PPO has partial seed results and takes too long"
    ),
}


def make_output_dir(out_root: Path, run_name: str) -> Path:
    # Create a timestamped result directory for one output record.
    out_dir = out_root.resolve() / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_outputs(out_dir: Path, run_name: str, metrics: dict, model_info: dict) -> tuple[Path, Path]:
    # Write the per-run comparison plot and compact metadata JSON.
    import matplotlib.pyplot as plt

    plot_path = out_dir / "comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, title, color in zip(
        axes,
        ("win_rate", "avg_reward", "avg_length"),
        ("Accuracy (Win Rate %)", "Average Episode Reward", "Average Episode Length"),
        ("#1f77b4", "#2ca02c", "#ff7f0e"),
    ):
        ax.bar([run_name], [metrics[metric]], color=color)
        ax.set_title(title)
    axes[0].set_ylabel("Percent")
    axes[0].set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    metadata_path = out_dir / "model_info.json"
    metadata_path.write_text(json.dumps(model_info, indent=2), encoding="utf-8")
    return plot_path, metadata_path


def maybe_reexec_in_preferred_venv():
    # Relaunch this script with the preferred virtual environment.
    if os.environ.get(REEXEC_GUARD_ENV) == "1":
        return
    if not PREFERRED_VENV_PYTHON.exists():
        return

    current = Path(sys.executable).resolve()
    preferred = PREFERRED_VENV_PYTHON.resolve()
    if str(current).lower() == str(preferred).lower():
        return

    print(f"Re-launching with venv interpreter: {preferred}")
    env = os.environ.copy()
    env[REEXEC_GUARD_ENV] = "1"
    completed = subprocess.run([str(preferred), str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
    raise SystemExit(completed.returncode)


maybe_reexec_in_preferred_venv()

import numpy as np
import torch
import torch.nn as nn
from procgen import ProcgenEnv
from torch.distributions.categorical import Categorical


class ResidualBlock(nn.Module):
    # Define the residual CNN block used by the policy network.
    def __init__(self, channels):
        # Initialize this object with the required layers or state.
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Run the module forward pass.
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    # Define one convolutional downsampling sequence for the policy network.
    def __init__(self, input_shape, out_channels):
        # Initialize this object with the required layers or state.
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(input_shape[0], out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(out_channels)
        self.res_block1 = ResidualBlock(out_channels)

    def get_output_shape(self):
        # Return the spatial output shape after this convolution sequence.
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

    def forward(self, x):
        # Run the module forward pass.
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x


class Agent(nn.Module):
    # Define the Clean PPO policy/value network used for evaluation.
    def __init__(self):
        # Initialize this object with the required layers or state.
        super().__init__()
        shape = (3, 64, 64)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.network = nn.Sequential(
            *conv_seqs,
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(shape[0] * shape[1] * shape[2], 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, 15)
        self.critic = nn.Linear(256, 1)

    def act(self, obs, sample_actions):
        # Choose either sampled or greedy actions from policy logits.
        hidden = self.network(obs.permute((0, 3, 1, 2)) / 255.0)
        logits = self.actor(hidden)
        if sample_actions:
            return Categorical(logits=logits).sample()
        return logits.argmax(dim=1)


def main():
    # Run the script entry point.
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="*", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "evaluation" / "results")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--start-level", type=int, default=50_000)
    parser.add_argument("--num-levels", type=int, default=0)
    parser.add_argument("--distribution-mode", choices=("easy", "hard"), default=None)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--eval-seed", type=int, default=None, help="Evaluate one seed only.")
    parser.add_argument(
        "--seed-timeout-seconds",
        type=int,
        default=DEFAULT_SEED_TIMEOUT_SECONDS,
        help="Wall-clock seconds allowed per eval seed; use 0 to disable.",
    )
    parser.add_argument("--argmax", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-evaluate runs even when results already exist.")
    args = parser.parse_args()
    args.eval_seeds = unique_eval_seeds([args.eval_seed] if args.eval_seed is not None else args.eval_seeds)

    checkpoints = args.checkpoints or discover_latest_checkpoints(args.runs_dir)
    if not checkpoints:
        raise FileNotFoundError("No clean_ppo_m*.pt or pb2_clean best_model.pt checkpoints found.")

    checkpoints, manually_skipped = skip_manual_runs(checkpoints)
    if manually_skipped:
        print("Manually skipped clean/PB2-clean models:")
        for run_name, reason in manually_skipped:
            print(f" - {run_name}: {reason}")
    if not checkpoints:
        print("No unskipped clean/PB2-clean models found.")
        return

    if not args.no_save and not args.force:
        checkpoints, skipped_evaluated = skip_already_evaluated(checkpoints, args.out_dir.resolve())
        if skipped_evaluated:
            print(f"Skipping {len(skipped_evaluated)} already evaluated clean/PB2-clean models.")
        if not checkpoints:
            print("No unevaluated clean/PB2-clean models found.")
            return

    for checkpoint_path in checkpoints:
        evaluate_checkpoint(checkpoint_path.resolve(), args)


def discover_latest_checkpoints(runs_dir: Path) -> list[Path]:
    # Find the latest Clean PPO/PB2 Clean checkpoint per game and mode.
    latest_by_source_env_mode: dict[tuple[str, str, str], tuple[int, Path]] = {}
    checkpoint_paths = [
        *runs_dir.glob("clean_ppo_m_*/clean_ppo_m*.pt"),
        *runs_dir.glob("pb2_clean_*/best_model.pt"),
    ]
    for checkpoint_path in sorted(checkpoint_paths):
        source = infer_source(checkpoint_path)
        env_name = infer_env_name(checkpoint_path)
        distribution_mode = infer_distribution_mode(checkpoint_path)
        run_id_text = infer_run_id(checkpoint_path)
        try:
            run_id = int(run_id_text)
        except ValueError:
            run_id = -1

        key = (source, env_name, distribution_mode)
        current = latest_by_source_env_mode.get(key)
        if current is None or run_id > current[0]:
            latest_by_source_env_mode[key] = (run_id, checkpoint_path)

    ordered_sources = ("clean_ppo_m", "pb2_clean")
    ordered_modes = ("easy", "hard")
    checkpoints = [
        latest_by_source_env_mode[(source, env_name, mode)][1]
        for source in ordered_sources
        for env_name in ORDERED_ENVS
        for mode in ordered_modes
        if (source, env_name, mode) in latest_by_source_env_mode
    ]
    remaining = [
        item[1]
        for key, item in latest_by_source_env_mode.items()
        if key[0] not in ordered_sources or key[1] not in ORDERED_ENVS or key[2] not in ordered_modes
    ]
    return checkpoints + sorted(remaining)


def skip_already_evaluated(checkpoints: list[Path], out_dir: Path) -> tuple[list[Path], list[str]]:
    # Remove checkpoints that already have average results.
    evaluated_names, evaluated_ids = find_already_evaluated_runs(out_dir)
    pending = []
    skipped = []
    for checkpoint_path in checkpoints:
        run_name = infer_run_name(checkpoint_path)
        run_id = infer_run_id(checkpoint_path)
        run_id_value = int(run_id) if run_id.isdigit() else None
        target = skipped if run_name in evaluated_names or run_id_value in evaluated_ids else pending
        target.append(run_name if target is skipped else checkpoint_path)
    return pending, skipped


def skip_manual_runs(checkpoints: list[Path]) -> tuple[list[Path], list[tuple[str, str]]]:
    # Remove checkpoints that are intentionally excluded from evaluation.
    pending = []
    skipped = []
    for checkpoint_path in checkpoints:
        run_name = infer_run_name(checkpoint_path)
        reason = MANUAL_SKIPPED_RUNS.get(run_name)
        (skipped if reason else pending).append((run_name, reason) if reason else checkpoint_path)
    return pending, skipped


def find_already_evaluated_runs(out_dir: Path) -> tuple[set[str], set[int]]:
    # Find runs that already have protocol-compatible average results.
    evaluated_names: set[str] = set()
    evaluated_ids: set[int] = set()
    if not out_dir.exists():
        return evaluated_names, evaluated_ids

    for metadata_path in out_dir.glob("*/model_info.json"):
        metadata = read_json(metadata_path)
        if metadata.get("evaluation_protocol") != EVALUATION_PROTOCOL:
            continue
        if metadata.get("evaluation_result") != "average":
            continue

        run_name = metadata.get("run_name")
        if isinstance(run_name, str) and run_name:
            evaluated_names.add(run_name)

        run_id = metadata.get("run_id")
        if isinstance(run_id, int):
            evaluated_ids.add(run_id)

    return evaluated_names, evaluated_ids


def unique_eval_seeds(seeds: list[int]) -> list[int]:
    # Deduplicate evaluation seeds while preserving order.
    unique = []
    seen = set()
    for seed in seeds:
        if seed in seen:
            continue
        seen.add(seed)
        unique.append(seed)
    return unique


def normalize_seed(seed) -> int | None:
    # Convert a stored training seed to an integer when possible.
    try:
        return int(seed)
    except (TypeError, ValueError):
        return None


def should_save_individual_seed(explicit_train_seed: int | None, seed: int, eval_seeds: list[int]) -> bool:
    # Decide whether this single-seed result should be saved.
    if len(eval_seeds) == 1:
        return True
    if seed == 1:
        return explicit_train_seed == 1
    return seed == INDIVIDUAL_REFERENCE_SEED


def seed_output_name(run_name: str, seed: int) -> str:
    # Build the output label for a single-seed evaluation.
    return f"{run_name}_evalseed{seed}"


def average_output_name(run_name: str, eval_seeds: list[int]) -> str:
    # Build the output label for a multi-seed average.
    seed_label = "_".join(str(seed) for seed in sorted(eval_seeds))
    return f"{run_name}_evalavg_{seed_label}"


def average_seed_metrics(seed_results: dict[int, dict]) -> dict:
    # Aggregate per-seed Clean PPO metrics into one weighted average.
    total_episodes = sum(metrics["episodes"] for metrics in seed_results.values())
    total_wins = sum(metrics["wins"] for metrics in seed_results.values())
    metrics = {
        "episodes": total_episodes,
        "requested_episodes": sum(metrics.get("requested_episodes", metrics["episodes"]) for metrics in seed_results.values()),
        "wins": total_wins,
        "win_rate": 100.0 * total_wins / total_episodes if total_episodes else 0.0,
        "avg_reward": weighted_metric_average(seed_results, "avg_reward"),
        "avg_length": weighted_metric_average(seed_results, "avg_length"),
    }
    summarize_average_runtime(metrics, seed_results)
    return metrics


def weighted_metric_average(seed_results: dict[int, dict], metric_name: str) -> float:
    # Compute an episode-weighted metric average across seeds.
    total_episodes = sum(metrics["episodes"] for metrics in seed_results.values())
    return 0.0 if total_episodes == 0 else sum(
        metrics[metric_name] * metrics["episodes"] for metrics in seed_results.values()
    ) / total_episodes


def add_runtime_note(model_info: dict, metrics: dict) -> None:
    # Attach timeout metadata to saved model info when needed.
    status = metrics.get("evaluation_status", "complete")
    if status != "complete":
        model_info["evaluation_status"] = status
        model_info["timeout_seconds"] = metrics.get("timeout_seconds")
        if metrics.get("note"):
            model_info["note"] = metrics["note"]


def summarize_average_runtime(metrics: dict, seed_results: dict[int, dict]) -> None:
    # Mark averaged results partial when any seed timed out.
    if all(seed_metrics.get("evaluation_status", "complete") == "complete" for seed_metrics in seed_results.values()):
        return

    metrics["evaluation_status"] = "partial"
    metrics["note"] = "One or more eval seeds timed out; averages use completed episodes only."
    timeout_values = [
        seed_metrics.get("timeout_seconds")
        for seed_metrics in seed_results.values()
        if seed_metrics.get("timeout_seconds") is not None
    ]
    if timeout_values:
        metrics["timeout_seconds"] = max(timeout_values)


def format_runtime_status(metrics: dict) -> str:
    # Format timeout status for console progress output.
    status = metrics.get("evaluation_status", "complete")
    return "" if status == "complete" else (
        f" | status={status} ({metrics['episodes']}/{metrics['requested_episodes']} episodes)"
    )


def evaluate_checkpoint(checkpoint_path: Path, args):
    # Evaluate one Clean PPO checkpoint across requested seeds.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    run_name = infer_run_name(checkpoint_path)
    source = infer_source(checkpoint_path)
    env_name = train_args.get("env_id") or infer_env_name(checkpoint_path)
    distribution_mode = args.distribution_mode or train_args.get("distribution_mode") or infer_distribution_mode(checkpoint_path)
    run_id = infer_run_id(checkpoint_path)

    agent = Agent().to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    print(
        f"Evaluating run id {run_id}: {checkpoint_path.name} | env={env_name} | mode={distribution_mode}",
        flush=True,
    )
    seed_results = {}
    explicit_train_seed = normalize_seed(train_args.get("seed"))
    for eval_seed in args.eval_seeds:
        print(f"  eval seed {eval_seed}", flush=True)
        metrics = evaluate(agent, env_name, distribution_mode, args, device, eval_seed)
        seed_results[eval_seed] = metrics
        print(
            f"seed {eval_seed}: win_rate={metrics['win_rate']:.2f}% | "
            f"avg_reward={metrics['avg_reward']:.3f} | avg_length={metrics['avg_length']:.1f}"
            f"{format_runtime_status(metrics)}",
            flush=True,
        )
        if not args.no_save and should_save_individual_seed(explicit_train_seed, eval_seed, args.eval_seeds):
            metadata_path = save_results(
                run_name=run_name,
                source=source,
                run_id=run_id,
                env_name=env_name,
                distribution_mode=distribution_mode,
                metrics=metrics,
                args=args,
                output_name=seed_output_name(run_name, eval_seed),
                eval_seed=eval_seed,
                eval_seeds=[eval_seed],
                evaluation_result="single_seed",
            )
            print(f"  {metadata_path}", flush=True)

    if len(seed_results) > 1:
        metrics = average_seed_metrics(seed_results)
        if not args.no_save:
            metadata_path = save_results(
                run_name=run_name,
                source=source,
                run_id=run_id,
                env_name=env_name,
                distribution_mode=distribution_mode,
                metrics=metrics,
                args=args,
                output_name=average_output_name(run_name, args.eval_seeds),
                eval_seed=None,
                eval_seeds=sorted(seed_results),
                evaluation_result="average",
            )
            print(f"  {metadata_path}", flush=True)
        print(
            f"avg {','.join(str(seed) for seed in sorted(seed_results))}: "
            f"win_rate={metrics['win_rate']:.2f}% | "
            f"avg_reward={metrics['avg_reward']:.3f} | avg_length={metrics['avg_length']:.1f}",
            flush=True,
        )


def evaluate(agent, env_name: str, distribution_mode: str, args, device, eval_seed: int):
    # Run Procgen episodes for one Clean PPO eval seed.
    started_at = time.monotonic()
    timed_out = False

    def seed_timed_out() -> bool:
        # Check whether this eval seed exceeded its wall-clock budget.
        return (
            args.seed_timeout_seconds is not None
            and args.seed_timeout_seconds > 0
            and time.monotonic() - started_at >= args.seed_timeout_seconds
        )

    env = ProcgenEnv(
        num_envs=args.num_envs,
        env_name=env_name,
        start_level=args.start_level,
        num_levels=args.num_levels,
        distribution_mode=distribution_mode,
        rand_seed=eval_seed,
    )
    obs = env.reset()["rgb"]
    ep_rewards = np.zeros(args.num_envs, dtype=np.float64)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int64)
    rewards = []
    lengths = []
    wins = 0
    completed = 0
    next_report = 10
    block_start = 1
    block_wins = 0
    block_count = 0
    print(f"  0% levels done (0/{args.episodes})", flush=True)

    try:
        while completed < args.episodes:
            if seed_timed_out():
                timed_out = True
                break

            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                actions = agent.act(obs_tensor, sample_actions=not args.argmax).cpu().numpy()

            obs_dict, step_rewards, dones, _infos = env.step(actions)
            obs = obs_dict["rgb"]
            ep_rewards += step_rewards
            ep_lengths += 1

            for index, done in enumerate(dones):
                if not done:
                    continue
                rewards.append(float(ep_rewards[index]))
                lengths.append(float(ep_lengths[index]))
                won = int(ep_rewards[index] >= 10.0)
                wins += won
                block_wins += won
                block_count += 1
                completed += 1
                ep_rewards[index] = 0.0
                ep_lengths[index] = 0
                if block_count == WIN_RATE_BLOCK_EPISODES or completed == args.episodes:
                    print(
                        f"  levels {block_start}-{completed}: win_rate={100.0 * block_wins / block_count:.2f}%",
                        flush=True,
                    )
                    block_start = completed + 1
                    block_wins = 0
                    block_count = 0
                percent = int(100 * completed / args.episodes)
                if percent >= next_report:
                    print(f"  {percent}% levels done ({completed}/{args.episodes})", flush=True)
                    next_report += 10
                if completed >= args.episodes:
                    break

            if seed_timed_out():
                timed_out = True
                break

        if timed_out:
            if block_count:
                print(
                    f"  levels {block_start}-{completed}: "
                    f"win_rate={100.0 * block_wins / block_count:.2f}% (partial before timeout)",
                    flush=True,
                )
            print(
                f"  seed timeout after {args.seed_timeout_seconds}s "
                f"({completed}/{args.episodes} completed episodes)",
                flush=True,
            )
    finally:
        env.close()

    status = "complete"
    note = None
    if timed_out:
        status = "timeout_partial" if completed else "timeout_no_episodes"
        note = (
            f"Timed out after {args.seed_timeout_seconds}s with "
            f"{completed}/{args.episodes} completed episodes."
        )
    return {
        "wins": wins,
        "episodes": completed,
        "requested_episodes": args.episodes,
        "win_rate": 100.0 * wins / completed if completed else 0.0,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        "evaluation_status": status,
        "timeout_seconds": args.seed_timeout_seconds if timed_out else None,
        "note": note,
    }


def save_results(
    *,
    run_name: str,
    source: str,
    run_id: str,
    env_name: str,
    distribution_mode: str,
    metrics: dict,
    args,
    output_name: str,
    eval_seed: int | None,
    eval_seeds: list[int],
    evaluation_result: str,
) -> Path:
    # Save one Clean PPO evaluation record and metadata JSON.
    out_dir = make_output_dir(args.out_dir, output_name)
    model_info = {
        "run_name": run_name,
        "run_id": int(run_id) if str(run_id).isdigit() else run_id,
        "family": "PB2 Clean PPO" if source == "pb2_clean" else "Clean PPO",
        "env_name": env_name,
        "distribution_mode": distribution_mode,
        "eval_seed": eval_seed,
        "eval_seeds": eval_seeds,
        "evaluation_result": evaluation_result,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "episodes": metrics["episodes"],
        "requested_episodes": metrics.get("requested_episodes", metrics["episodes"]),
        "wins": metrics["wins"],
        "win_rate": metrics["win_rate"],
        "avg_reward": metrics["avg_reward"],
        "avg_length": metrics["avg_length"],
    }
    add_runtime_note(model_info, metrics)
    _, metadata_path = write_outputs(out_dir, output_name, metrics, model_info)
    return metadata_path


def read_json(path: Path) -> dict:
    # Read JSON from disk, returning an empty object when absent.
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def infer_run_name(path: Path) -> str:
    # Infer the canonical run name from a checkpoint path.
    if path.parent.name.startswith(("clean_ppo_m_", "pb2_clean_")):
        return path.parent.name
    return path.stem


def infer_source(path: Path) -> str:
    # Infer whether the checkpoint is baseline Clean PPO or PB2 Clean PPO.
    return "pb2_clean" if infer_run_name(path).startswith("pb2_clean_") else "clean_ppo_m"


def infer_run_id(path: Path) -> str:
    # Extract the run id from a Clean PPO checkpoint path.
    match = re.search(r"_(\d+)(?:\.pt)?$", infer_run_name(path))
    return match.group(1) if match else "unknown"


def infer_env_name(path: Path) -> str:
    # Infer the Procgen environment from a checkpoint path.
    text = f"{path.parent.name}_{path.name}"
    for env_name in ORDERED_ENVS:
        if env_name in text:
            return env_name
    return "coinrun"


def infer_distribution_mode(path: Path) -> str:
    # Infer easy/hard mode from a checkpoint path.
    name = f"_{path.parent.name}_{path.stem}_"
    if "_H_" in name:
        return "hard"
    return "easy"


if __name__ == "__main__":
    main()
