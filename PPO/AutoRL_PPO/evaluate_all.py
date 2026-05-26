import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFERRED_VENV_PYTHON = Path(
    r"C:\Users\danim\PycharmProjects\GroupProject\.venv\Scripts\python.exe"
)
BASE_REQUIRED_PACKAGES = {
    "gymnasium": "gymnasium>=0.29,<1.3",
    "matplotlib": "matplotlib",
    "numpy": "numpy<2",
    "procgen": "procgen-mirror==0.10.7",
    "sb3_contrib": "sb3-contrib==2.7.1",
    "stable_baselines3": "stable-baselines3[extra]==2.7.1",
}
WIN_RATE_BLOCK_EPISODES = 100
DEFAULT_EVAL_SEEDS = (1, 10, 2, 3)
EVALUATION_PROTOCOL = "eval_seeds_1_2_3_10"
INDIVIDUAL_REFERENCE_SEED = 10
DEFAULT_SEED_TIMEOUT_SECONDS = 7 * 60


@dataclass(frozen=True)
class RunSpec:
    # Store metadata needed to evaluate one saved model.
    run_name: str
    run_id: int
    family: str
    algorithm: str
    env_name: str
    model_path: Path
    explicit_train_seed: int | None = None


@dataclass(frozen=True)
class SkippedRun:
    # Store a run folder that cannot be evaluated and why.
    run_name: str
    run_id: int | None
    reason: str


def prepare_environment(script_path: Path, reexec_guard_env: str) -> None:
    # Re-enter the preferred venv and ensure required packages exist.
    maybe_reexec_in_preferred_venv(script_path, reexec_guard_env)
    ensure_requirements_installed(BASE_REQUIRED_PACKAGES)


def maybe_reexec_in_preferred_venv(script_path: Path, reexec_guard_env: str) -> None:
    # Relaunch this script with the preferred virtual environment.
    if os.environ.get(reexec_guard_env) == "1":
        return
    if not PREFERRED_VENV_PYTHON.exists():
        return

    current = Path(sys.executable).resolve()
    preferred = PREFERRED_VENV_PYTHON.resolve()
    if str(current).lower() == str(preferred).lower():
        return

    print(f"Re-launching with venv interpreter: {preferred}")
    env = os.environ.copy()
    env[reexec_guard_env] = "1"
    cmd = [str(preferred), str(script_path.resolve()), *sys.argv[1:]]
    completed = subprocess.run(cmd, env=env)
    raise SystemExit(completed.returncode)


def ensure_requirements_installed(required_packages: dict[str, str]) -> None:
    # Install any missing evaluator dependencies.
    missing = [name for name in required_packages if importlib.util.find_spec(name) is None]
    if not missing:
        return

    print(f"Missing packages in venv: {', '.join(missing)}")
    packages = [required_packages[name] for name in missing]
    completed = subprocess.run([sys.executable, "-m", "pip", "install", *packages])
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def make_output_dir(out_root: Path, run_name: str) -> Path:
    # Create a timestamped result directory for one output record.
    out_dir = out_root.resolve() / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def collect_run_specs(runs_dir: Path) -> tuple[list[RunSpec], list[SkippedRun]]:
    # Discover evaluable non-Clean saved-model runs.
    specs: list[RunSpec] = []
    skipped: list[SkippedRun] = []
    for path in sorted(runs_dir.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue

        spec, reason = scan_run_dir(path)
        if spec is not None:
            specs.append(spec)
        elif reason is not None:
            skipped.append(reason)

    specs.sort(key=lambda spec: (spec.env_name == "heist", spec.run_id))
    skipped.sort(key=lambda item: (item.run_id is None, item.run_id if item.run_id is not None else -1, item.run_name))
    return specs, skipped


def scan_run_dir(run_dir: Path) -> tuple[RunSpec | None, SkippedRun | None]:
    # Convert one run directory into an evaluation spec or skip reason.
    run_name = run_dir.name
    run_id = extract_run_id(run_name)
    if run_id is None:
        return None, None

    if run_name.startswith("asha_"):
        model_path = run_dir / "best_model.zip"
        if not model_path.exists():
            return None, SkippedRun(run_name, run_id, "missing best_model.zip")
        algorithm = resolve_asha_algorithm(run_dir)
        return make_spec(run_dir, run_id, "ASHA", algorithm, resolve_asha_env_name(run_dir), model_path), None

    if run_name.startswith("pb2_clean_"):
        return None, SkippedRun(run_name, run_id, "Clean PPO .pt checkpoint; use evaluation/evaluate_clean_ppo.py")

    if run_name.startswith("pb2_"):
        model_path = resolve_pb2_best_model_path(run_dir)
        if model_path is None:
            return None, SkippedRun(run_name, run_id, "missing best_model.zip or available PB2 member model")
        algorithm = resolve_pb2_algorithm(run_dir)
        return make_spec(run_dir, run_id, "PB2", algorithm, resolve_pb2_env_name(run_dir), model_path), None

    single_model_specs = (
        ("recurrentppo_", "recurrentppo", "Recurrent PPO", "coinrun", ["lstm.zip"], None),
        ("main_impala_coinrun_", "impala", "IMPALA-style PPO", "coinrun", ["impala_coinrun.zip"], 1),
        ("main_impala_heist_", "impala", "IMPALA-style PPO", "heist", ["impala_heist.zip"], 1),
        ("main_impala_starpilot_", "impala", "IMPALA-style PPO", "starpilot", ["impala_starpilot.zip"], 1),
        ("ppo_heist_", "ppo", "PPO", "heist", ["ppo_heist.zip"], None),
        ("ppo_starpilot_", "ppo", "PPO", "starpilot", ["ppo_starpilot.zip"], None),
        ("ppo_", "ppo", "PPO", "coinrun", ["combinedv2.zip", "ppo.zip"], None),
    )

    for prefix, algorithm, family, env_name, preferred_names, explicit_train_seed in single_model_specs:
        if not run_name.startswith(prefix):
            continue

        model_path = find_run_model(run_dir, preferred_names)
        if model_path is None:
            return None, SkippedRun(
                run_name,
                run_id,
                f"missing model file ({', '.join(preferred_names)})",
            )

        return make_spec(run_dir, run_id, family, algorithm, env_name, model_path, explicit_train_seed), None

    return None, SkippedRun(run_name, run_id, "unrecognized run type")


def make_spec(
    run_dir: Path,
    run_id: int,
    family: str,
    algorithm: str,
    env_name: str,
    model_path: Path,
    explicit_train_seed: int | None = None,
) -> RunSpec:
    # Create a normalized run specification for evaluation.
    if explicit_train_seed is None and algorithm == "impala":
        explicit_train_seed = 1
    return RunSpec(run_dir.name, run_id, family, algorithm, env_name, model_path.resolve(), explicit_train_seed)


def extract_run_id(name: str) -> int | None:
    # Extract the numeric run id from a run directory name.
    match = re.search(r"_(\d+)$", name)
    return int(match.group(1)) if match else None


def find_run_model(run_dir: Path, preferred_model_names: list[str]) -> Path | None:
    # Find the preferred model file inside a run directory.
    for model_name in preferred_model_names:
        model_path = run_dir / model_name
        if model_path.exists():
            return model_path

    models = sorted(run_dir.glob("*.zip"))
    return models[0] if len(models) == 1 else None


def resolve_pb2_best_model_path(run_dir: Path) -> Path | None:
    # Find the best model checkpoint for a PB2 run.
    best_model_path = run_dir / "best_model.zip"
    if best_model_path.exists():
        return best_model_path.resolve()

    best_result_path = run_dir / "best_result.json"
    if best_result_path.exists():
        best_result = json.loads(best_result_path.read_text(encoding="utf-8"))
        model_path = resolve_local_model_path(run_dir, str(best_result.get("model_path", "") or ""))
        if model_path is not None:
            return model_path.resolve()

    best_score = None
    best_path = None
    for result_path in run_dir.glob("member_*/round_*/result.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        score = result.get("score")
        if not isinstance(score, (int, float)):
            continue

        model_path = resolve_local_model_path(run_dir, str(result.get("model_path", "") or ""))
        if model_path is None:
            continue

        if best_score is None or float(score) > best_score:
            best_score = float(score)
            best_path = model_path

    return best_path.resolve() if best_path is not None else None


def resolve_local_model_path(run_dir: Path, model_path: str) -> Path | None:
    # Resolve a stored model path back inside its run directory.
    if not model_path:
        return None

    path = Path(model_path)
    if path.exists():
        return path

    parts = list(path.parts)
    if run_dir.name in parts:
        relative_parts = parts[parts.index(run_dir.name) + 1 :]
        candidate = run_dir.joinpath(*relative_parts)
        if candidate.exists():
            return candidate

    return None


def resolve_pb2_env_name(run_dir: Path) -> str:
    # Infer the environment name for a PB2 run.
    return state_value(run_dir, "pb2_state.json", "env_name") or env_from_name(run_dir.name)


def resolve_pb2_algorithm(run_dir: Path) -> str:
    # Infer whether a PB2 run uses PPO or IMPALA-style PPO.
    state_path = run_dir / "pb2_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        algorithm = state.get("algorithm")
        if algorithm:
            return algorithm
        if algorithm_from_base_module(state.get("base_module", "")) == "impala":
            return "impala"

    if run_dir.name.startswith("pb2_impala_"):
        return "impala"
    return "ppo"


def resolve_asha_algorithm(run_dir: Path) -> str:
    # Infer whether an ASHA run uses PPO, recurrent PPO, or IMPALA.
    state_path = run_dir / "asha_state.json"
    if not state_path.exists():
        return "ppo"

    state = json.loads(state_path.read_text(encoding="utf-8"))
    algorithm = state.get("algorithm")
    if algorithm:
        return algorithm
    if algorithm_from_base_module(state.get("base_module", "")) == "impala":
        return "impala"
    best = state.get("best", {})
    model_path = str(best.get("model_path", "") or "")
    if model_path.endswith("lstm.zip"):
        return "recurrentppo"

    for trial in state.get("trials", []):
        trial_model_path = str(trial.get("model_path", "") or "")
        if trial_model_path.endswith("lstm.zip"):
            return "recurrentppo"

    return "ppo"


def resolve_asha_env_name(run_dir: Path) -> str:
    # Infer the environment name for an ASHA run.
    return state_value(run_dir, "asha_state.json", "env_name") or env_from_name(run_dir.name)


def state_value(run_dir: Path, state_name: str, key: str):
    # Read one value from a search state file when present.
    state_path = run_dir / state_name
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get(key):
            return state[key]
    return None


def env_from_name(run_name: str) -> str:
    # Infer the Procgen environment from a run name.
    if "_heist_" in run_name:
        return "heist"
    if "_starpilot_" in run_name:
        return "starpilot"
    return "coinrun"


def algorithm_from_base_module(base_module: str) -> str:
    # Map a base module string to the evaluator algorithm label.
    return "impala" if "impala" in str(base_module).lower() else "ppo"


def evaluate_run(
    spec: RunSpec,
    *,
    episodes: int,
    num_envs: int,
    start_level: int,
    num_levels: int,
    eval_seed: int,
    deterministic: bool,
    seed_timeout_seconds: int | None,
) -> tuple[dict, dict]:
    # Evaluate one run spec on one seed and build metadata.
    metrics = evaluate_model(
        model_path=spec.model_path,
        algorithm=spec.algorithm,
        episodes=episodes,
        num_envs=num_envs,
        start_level=start_level,
        num_levels=num_levels,
        eval_seed=eval_seed,
        deterministic=deterministic,
        env_name=spec.env_name,
        seed_timeout_seconds=seed_timeout_seconds,
    )

    model_info = {
        "run_name": spec.run_name,
        "run_id": spec.run_id,
        "family": spec.family,
        "env_name": spec.env_name,
        "eval_seed": eval_seed,
        "eval_seeds": [eval_seed],
        "evaluation_result": "single_seed",
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "episodes": metrics["episodes"],
        "requested_episodes": metrics["requested_episodes"],
        "wins": metrics["wins"],
        "win_rate": metrics["win_rate"],
        "avg_reward": metrics["avg_reward"],
        "avg_length": metrics["avg_length"],
    }
    add_runtime_note(model_info, metrics)

    return metrics, model_info


def save_run_outputs(out_root: Path, output_name: str, metrics: dict, model_info: dict) -> Path:
    # Write comparison plot and metadata for one evaluated result.
    out_dir = make_output_dir(out_root, output_name)
    _, metadata_path = write_outputs(out_dir, output_name, metrics, model_info)
    return metadata_path


def should_save_individual_seed(spec: RunSpec, seed: int, eval_seeds: list[int]) -> bool:
    # Decide whether this single-seed result should be saved.
    if len(eval_seeds) == 1:
        return True
    if seed == 1:
        return spec.explicit_train_seed == 1
    return seed == INDIVIDUAL_REFERENCE_SEED


def seed_output_name(run_name: str, seed: int) -> str:
    # Build the output label for a single-seed evaluation.
    return f"{run_name}_evalseed{seed}"


def average_output_name(run_name: str, eval_seeds: list[int]) -> str:
    # Build the output label for a multi-seed average.
    seed_label = "_".join(str(seed) for seed in sorted(eval_seeds))
    return f"{run_name}_evalavg_{seed_label}"


def average_seed_results(seed_results: dict[int, tuple[dict, dict]]) -> tuple[dict, dict]:
    # Aggregate per-seed metrics into one weighted average result.
    first_seed = next(iter(seed_results))
    first_info = seed_results[first_seed][1]
    total_episodes = sum(metrics["episodes"] for metrics, _ in seed_results.values())
    total_wins = sum(metrics["wins"] for metrics, _ in seed_results.values())
    metrics = {
        "episodes": total_episodes,
        "requested_episodes": sum(metrics.get("requested_episodes", metrics["episodes"]) for metrics, _ in seed_results.values()),
        "wins": total_wins,
        "win_rate": 100.0 * total_wins / total_episodes if total_episodes else 0.0,
        "avg_reward": weighted_metric_average(seed_results, "avg_reward"),
        "avg_length": weighted_metric_average(seed_results, "avg_length"),
    }
    model_info = dict(first_info)
    model_info.update(
        {
            "eval_seed": None,
            "eval_seeds": sorted(seed_results),
            "evaluation_result": "average",
            "episodes": metrics["episodes"],
            "requested_episodes": metrics["requested_episodes"],
            "wins": metrics["wins"],
            "win_rate": metrics["win_rate"],
            "avg_reward": metrics["avg_reward"],
            "avg_length": metrics["avg_length"],
        }
    )
    summarize_average_runtime(model_info, seed_results)
    return metrics, model_info


def weighted_metric_average(seed_results: dict[int, tuple[dict, dict]], metric_name: str) -> float:
    # Compute an episode-weighted metric average across seeds.
    total_episodes = sum(metrics["episodes"] for metrics, _ in seed_results.values())
    return 0.0 if total_episodes == 0 else sum(
        metrics[metric_name] * metrics["episodes"] for metrics, _ in seed_results.values()
    ) / total_episodes


def add_runtime_note(model_info: dict, metrics: dict) -> None:
    # Attach timeout metadata to saved model info when needed.
    status = metrics.get("evaluation_status", "complete")
    if status != "complete":
        model_info["evaluation_status"] = status
        model_info["timeout_seconds"] = metrics.get("timeout_seconds")
        if metrics.get("note"):
            model_info["note"] = metrics["note"]


def summarize_average_runtime(model_info: dict, seed_results: dict[int, tuple[dict, dict]]) -> None:
    # Mark averaged results partial when any seed timed out.
    if all(metrics.get("evaluation_status", "complete") == "complete" for metrics, _ in seed_results.values()):
        return

    model_info["evaluation_status"] = "partial"
    model_info["note"] = "One or more eval seeds timed out; averages use completed episodes only."
    timeout_values = [
        metrics.get("timeout_seconds")
        for metrics, _ in seed_results.values()
        if metrics.get("timeout_seconds") is not None
    ]
    if timeout_values:
        model_info["timeout_seconds"] = max(timeout_values)


def format_runtime_status(metrics: dict) -> str:
    # Format timeout status for console progress output.
    status = metrics.get("evaluation_status", "complete")
    return "" if status == "complete" else (
        f" | status={status} ({metrics['episodes']}/{metrics['requested_episodes']} episodes)"
    )


def evaluate_model(
    model_path: Path,
    algorithm: str,
    episodes: int,
    num_envs: int,
    start_level: int,
    num_levels: int,
    eval_seed: int,
    deterministic: bool,
    env_name: str = "coinrun",
    win_reward: float = 10.0,
    seed_timeout_seconds: int | None = DEFAULT_SEED_TIMEOUT_SECONDS,
) -> dict:
    # Evaluate a model and summarize win/reward/length metrics.
    import numpy as np

    started_at = time.monotonic()
    timed_out = False
    env = None
    completed = 0
    wins = 0
    rewards: list[float] = []
    lengths: list[float] = []

    def seed_timed_out() -> bool:
        # Check whether this eval seed exceeded its wall-clock budget.
        return (
            seed_timeout_seconds is not None
            and seed_timeout_seconds > 0
            and time.monotonic() - started_at >= seed_timeout_seconds
        )

    try:
        env = make_procgen_env(num_envs, start_level, num_levels, eval_seed, env_name)
        model = load_model(model_path, algorithm)
        obs = env.reset()
        state = None
        episode_starts = np.ones((num_envs,), dtype=bool) if algorithm == "recurrentppo" else None
        block_start = 1
        block_wins = 0
        block_count = 0

        while completed < episodes:
            if seed_timed_out():
                timed_out = True
                break

            if algorithm == "recurrentppo":
                actions, state = model.predict(
                    obs,
                    state=state,
                    episode_start=episode_starts,
                    deterministic=deterministic,
                )
            else:
                actions, _ = model.predict(obs, deterministic=deterministic)

            obs, _, dones, infos = env.step(actions)
            done_flags = np.asarray(dones, dtype=bool)
            if algorithm == "recurrentppo":
                episode_starts = done_flags

            for idx, done in enumerate(done_flags):
                if not done:
                    continue

                episode_info = infos[idx].get("episode", {})
                reward = float(episode_info.get("r", 0.0))
                length = float(episode_info.get("l", 0.0))
                rewards.append(reward)
                lengths.append(length)
                won = int(reward >= win_reward)
                wins += won
                block_wins += won
                block_count += 1
                completed += 1
                if block_count == WIN_RATE_BLOCK_EPISODES or completed == episodes:
                    print(
                        f"  levels {block_start}-{completed}: win_rate={100.0 * block_wins / block_count:.2f}%",
                        flush=True,
                    )
                    block_start = completed + 1
                    block_wins = 0
                    block_count = 0

                if completed >= episodes:
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
                f"  seed timeout after {seed_timeout_seconds}s "
                f"({completed}/{episodes} completed episodes)",
                flush=True,
            )
    finally:
        if env is not None:
            env.close()

    evaluated_episodes = completed
    status = "complete"
    note = None
    if timed_out:
        status = "timeout_partial" if evaluated_episodes else "timeout_no_episodes"
        note = (
            f"Timed out after {seed_timeout_seconds}s with "
            f"{evaluated_episodes}/{episodes} completed episodes."
        )
    metrics = {
        "episodes": evaluated_episodes,
        "requested_episodes": episodes,
        "wins": wins,
        "win_rate": 100.0 * wins / evaluated_episodes if evaluated_episodes else 0.0,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        "evaluation_status": status,
        "timeout_seconds": seed_timeout_seconds if timed_out else None,
        "note": note,
    }
    return metrics


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


def make_procgen_env(num_envs: int, start_level: int, num_levels: int, eval_seed: int, env_name: str):
    # Build a Procgen vector environment for training or evaluation.
    import gymnasium as gym
    import numpy as np
    from procgen import ProcgenEnv
    from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor

    class ProcgenRGBWrapper(VecEnvWrapper):
        # Expose Procgen RGB observations through the SB3 VecEnv API.
        def __init__(self, venv):
            # Initialize this object with the required layers or state.
            super().__init__(venv)
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
            self.action_space = gym.spaces.Discrete(15)

        def reset(self):
            # Reset the wrapped Procgen environment and return RGB frames.
            obs = self.venv.reset()
            return obs["rgb"]

        def step_async(self, actions):
            # Forward asynchronous actions to the wrapped vector environment.
            self.venv.step_async(actions)

        def step_wait(self):
            # Return RGB observations plus rewards, dones, and infos.
            obs, rewards, dones, infos = self.venv.step_wait()
            return obs["rgb"], rewards, dones, infos

    env = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        start_level=start_level,
        num_levels=num_levels,
        distribution_mode="hard",
        rand_seed=eval_seed,
    )
    env = ProcgenRGBWrapper(env)
    return VecMonitor(env)


def load_model(model_path: Path, algorithm: str):
    # Load the correct SB3 model class for this algorithm.
    if algorithm == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(str(model_path))
    if algorithm == "impala":
        from stable_baselines3 import PPO

        return PPO.load(str(model_path), custom_objects={"policy_kwargs": load_impala_policy_kwargs()})
    if algorithm == "recurrentppo":
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(str(model_path))
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def load_impala_policy_kwargs() -> dict:
    # Return the custom IMPALA policy kwargs needed for loading.
    from main_impala import ImpalaCNN

    return {
        "features_extractor_class": ImpalaCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }


REEXEC_GUARD_ENV = "EVALUATE_ALL_REEXEC_DONE"


def main() -> None:
    # Run the script entry point.
    parser = argparse.ArgumentParser(description="Evaluate every saved model across all run types.")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "evaluation" / "results")
    parser.add_argument("--force", action="store_true", help="Re-evaluate runs even when results already exist.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--start-level", type=int, default=50_000)
    parser.add_argument("--num-levels", type=int, default=0)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--eval-seed", type=int, default=None, help="Evaluate one seed only.")
    parser.add_argument(
        "--seed-timeout-seconds",
        type=int,
        default=DEFAULT_SEED_TIMEOUT_SECONDS,
        help="Wall-clock seconds allowed per eval seed; use 0 to disable.",
    )
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    eval_seeds = unique_eval_seeds([args.eval_seed] if args.eval_seed is not None else args.eval_seeds)

    prepare_environment(Path(__file__), REEXEC_GUARD_ENV)

    specs, skipped = collect_run_specs(args.runs_dir.resolve())
    if not specs:
        raise FileNotFoundError(f"No evaluable runs found in {args.runs_dir.resolve()}")

    already_evaluated_run_names, already_evaluated_run_ids = find_already_evaluated_runs(args.out_dir.resolve())
    skipped_evaluated = []
    if not args.force and (already_evaluated_run_names or already_evaluated_run_ids):
        pending_specs = []
        for spec in specs:
            target = skipped_evaluated if spec.run_name in already_evaluated_run_names or spec.run_id in already_evaluated_run_ids else pending_specs
            target.append(spec.run_name if target is skipped_evaluated else spec)
        specs = pending_specs

    failures: list[tuple[str, str]] = []
    if skipped_evaluated:
        print(f"Skipping {len(skipped_evaluated)} already evaluated models.")

    if not specs:
        print("No unevaluated saved models found.")
        if skipped:
            print("Skipped run folders:")
            for item in skipped:
                print(f" - {item.run_name}: {item.reason}")
        return

    print(f"Evaluating {len(specs)} saved models...")
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] {spec.run_name}")
        try:
            seed_results = {}
            for eval_seed in eval_seeds:
                print(f"  eval seed {eval_seed}", flush=True)
                metrics, model_info = evaluate_run(
                    spec,
                    episodes=args.episodes,
                    num_envs=args.num_envs,
                    start_level=args.start_level,
                    num_levels=args.num_levels,
                    eval_seed=eval_seed,
                    deterministic=not args.stochastic,
                    seed_timeout_seconds=args.seed_timeout_seconds,
                )
                seed_results[eval_seed] = (metrics, model_info)
                print(
                    f"  seed {eval_seed}: win_rate={metrics['win_rate']:.2f}% | "
                    f"avg_reward={metrics['avg_reward']:.3f} | avg_length={metrics['avg_length']:.1f}"
                    f"{format_runtime_status(metrics)}"
                )
                if should_save_individual_seed(spec, eval_seed, eval_seeds):
                    metadata_path = save_run_outputs(
                        args.out_dir.resolve(),
                        seed_output_name(spec.run_name, eval_seed),
                        metrics,
                        model_info,
                    )
                    print(f"  {metadata_path}")

            if len(seed_results) > 1:
                metrics, model_info = average_seed_results(seed_results)
                metadata_path = save_run_outputs(
                    args.out_dir.resolve(),
                    average_output_name(spec.run_name, eval_seeds),
                    metrics,
                    model_info,
                )
                print(
                    f"  avg {','.join(str(seed) for seed in sorted(seed_results))}: "
                    f"win_rate={metrics['win_rate']:.2f}% | "
                    f"avg_reward={metrics['avg_reward']:.3f} | avg_length={metrics['avg_length']:.1f}"
                )
                print(f"  {metadata_path}")
        except Exception as exc:
            failures.append((spec.run_name, str(exc)))
            print(f"  FAILED: {exc}")

    if skipped:
        print("Skipped run folders:")
        for item in skipped:
            print(f" - {item.run_name}: {item.reason}")

    if failures:
        print("Evaluation failures:")
        for run_name, error in failures:
            print(f" - {run_name}: {error}")
        raise SystemExit(1)


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


def find_already_evaluated_runs(out_dir: Path) -> tuple[set[str], set[int]]:
    # Find runs that already have protocol-compatible average results.
    evaluated_names: set[str] = set()
    evaluated_ids: set[int] = set()
    if not out_dir.exists():
        return evaluated_names, evaluated_ids

    for metadata_path in out_dir.glob("*/model_info.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

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


if __name__ == "__main__":
    main()
