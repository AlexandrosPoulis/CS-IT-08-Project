import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


MAX_GPUS = 4
STAGE_TIMESTEPS = [1_000_000, 3_000_000, 12_000_000, 48_000_000]
PARAM_SPACE = {
    "learning_rate": {"type": "log_float", "min": 1e-4, "max": 8e-4},
    "n_steps": {"type": "choice", "values": [128, 256]},
    "batch_size": {"type": "choice", "values": [4096, 8192]},
    "n_epochs": {"type": "choice", "values": [3, 6, 10]},
    "gamma": {"type": "choice", "values": [0.99, 0.995, 0.999]},
    "gae_lambda": {"type": "choice", "values": [0.95, 0.97, 0.99]},
    "ent_coef": {"type": "log_float", "min": 5e-5, "max": 5e-4},
    "clip_range": {"type": "choice", "values": [0.1, 0.2, 0.3]},
    "vf_coef": {"type": "choice", "values": [0.5, 0.75, 1.0]},
    "max_grad_norm": {"type": "choice", "values": [0.5, 1.0]},
}


def parse_args() -> argparse.Namespace:
    # Parse command-line options for this script.
    parser = argparse.ArgumentParser(description="ASHA search wrapper for ppo.py")
    parser.add_argument("--output_root", default="asha_runs")
    parser.add_argument("--base_module", default="ppo")
    parser.add_argument("--env_name", default="coinrun")
    parser.add_argument("--max_trials", type=int, default=8)
    parser.add_argument("--parallel_trials", type=int, default=MAX_GPUS)
    parser.add_argument("--gpus", type=int, default=MAX_GPUS)
    parser.add_argument("--master_seed", type=int, default=1234)
    parser.add_argument("--poll_seconds", type=int, default=5)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--trial_dir", default="")
    parser.add_argument("--stage_index", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=0)
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--params_json", default="")
    return parser.parse_args()


def sample_params(rng: random.Random) -> dict:
    # Sample an initial hyperparameter configuration.
    params = {}
    for name, spec in PARAM_SPACE.items():
        if spec["type"] == "choice":
            params[name] = rng.choice(spec["values"])
        else:
            params[name] = 10 ** rng.uniform(math.log10(spec["min"]), math.log10(spec["max"]))
    return params


def load_or_create_state(args: argparse.Namespace, output_root: Path) -> tuple[dict, Path]:
    # Resume an existing search state or initialize a new one.
    state_path = output_root / "asha_state.json"
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8")), state_path

    rng = random.Random(args.master_seed)
    state = {
        "stage_timesteps": STAGE_TIMESTEPS,
        "base_module": args.base_module,
        "env_name": args.env_name,
        "algorithm": "ppo",
        "trials": [],
        "best": {"trial_id": "", "score": None, "params": {}, "model_path": ""},
    }
    for i in range(args.max_trials):
        state["trials"].append(
            {
                "trial_id": f"trial_{i:03d}",
                "params": sample_params(rng),
                "status": "pending",
                "stages": [],
            }
        )
    save_state(state, state_path)
    return state, state_path


def save_state(state: dict, state_path: Path) -> None:
    # Persist the search manager state to disk.
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def gpu_pool(args: argparse.Namespace) -> list[str]:
    # Build the ASHA GPU allocation pool.
    limit = min(MAX_GPUS, max(0, args.parallel_trials), max(0, args.gpus))
    if limit == 0:
        return []

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
        return devices[:limit]

    return [str(index) for index in range(limit)]


def stage_result(trial: dict, stage_index: int) -> dict | None:
    # Load the completed result for one ASHA trial stage.
    matches = [stage for stage in trial["stages"] if stage["stage_index"] == stage_index and stage["status"] == "done"]
    return matches[-1] if matches else None


def render_worker(trial_dir: Path, params: dict, timesteps: int, base_module: str, env_name: str) -> Path:
    # Write a worker script specialized for one ASHA trial stage.
    worker_path = trial_dir / "worker_ppo.py"
    base_dir = Path(__file__).resolve().parent
    vf_coef = params.get("vf_coef", 0.75)
    worker_path.write_text(
        f"""import importlib
import json
import sys
from pathlib import Path

import numpy as np
from procgen import ProcgenEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecMonitor

sys.path.insert(0, r"{base_dir}")
base = importlib.import_module("{base_module}")

trial_dir = Path(r"{trial_dir}")
trial_dir.mkdir(parents=True, exist_ok=True)
(trial_dir / "tensorboard").mkdir(exist_ok=True)
(trial_dir / "eval").mkdir(exist_ok=True)

train_num_envs = 64
eval_num_envs = 16

env = ProcgenEnv(num_envs=train_num_envs, env_name="{env_name}", start_level=0, num_levels=0, distribution_mode="hard")
env = base.ProcgenRGBWrapper(env)
env = VecMonitor(env)

eval_env = ProcgenEnv(
    num_envs=eval_num_envs,
    env_name="{env_name}",
    start_level=base.eval_start_level,
    num_levels=base.eval_num_levels,
    distribution_mode="hard",
)
eval_env = base.ProcgenRGBWrapper(eval_env)
eval_env = VecMonitor(eval_env)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate={params["learning_rate"]},
    n_steps={params["n_steps"]},
    batch_size={params["batch_size"]},
    n_epochs={params["n_epochs"]},
    gamma={params["gamma"]},
    gae_lambda={params["gae_lambda"]},
    ent_coef={params["ent_coef"]},
    clip_range={params["clip_range"]},
    vf_coef={vf_coef},
    max_grad_norm={params["max_grad_norm"]},
    device="cuda",
    tensorboard_log=str(trial_dir / "tensorboard"),
)

callback = EvalCallback(
    eval_env,
    best_model_save_path=str(trial_dir / "best"),
    log_path=str(trial_dir / "eval"),
    eval_freq=max({timesteps} // 10 // train_num_envs, 1),
    n_eval_episodes=16,
    deterministic=True,
)

model.learn(total_timesteps={timesteps}, progress_bar=True, callback=callback)
model.save(str(trial_dir / "ppo"))

results = np.load(trial_dir / "eval" / "evaluations.npz", allow_pickle=True)["results"]
means = [float(np.mean(x)) for x in results]
summary = {{
    "score": max(means) if means else None,
    "last_score": means[-1] if means else None,
    "model_path": str((trial_dir / "ppo.zip").resolve()),
    "best_model_path": str((trial_dir / "best" / "best_model.zip").resolve()),
    "params": {json.dumps(params)},
}}
(trial_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
""",
        encoding="utf-8",
    )
    return worker_path


def run_worker(args: argparse.Namespace) -> None:
    # Train and evaluate one worker segment, then write its result.
    trial_dir = Path(args.trial_dir)
    params = json.loads(args.params_json)
    worker_path = render_worker(trial_dir, params, args.timesteps, args.base_module, args.env_name)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    result = subprocess.run([sys.executable, str(worker_path)], env=env, check=False)
    raise SystemExit(result.returncode)


def launch_trial(trial: dict, output_root: Path, stage_index: int, gpu_id: str) -> dict:
    # Start one ASHA trial stage as a subprocess.
    trial_dir = output_root / trial["trial_id"]
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "config.json").write_text(json.dumps(trial["params"], indent=2), encoding="utf-8")
    log_path = trial_dir / f"stage_{stage_index:02d}.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--trial_dir",
        str(trial_dir),
        "--base_module",
        args.base_module,
        "--env_name",
        args.env_name,
        "--stage_index",
        str(stage_index),
        "--timesteps",
        str(STAGE_TIMESTEPS[stage_index]),
        "--gpu_id",
        gpu_id,
        "--params_json",
        json.dumps(trial["params"]),
    ]
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    return {"trial": trial, "process": process, "log_file": log_file, "gpu_id": gpu_id, "trial_dir": trial_dir}


def collect_trial(handle: dict, state: dict, state_path: Path, stage_index: int, output_root: Path) -> None:
    # Record a completed ASHA trial stage and update promotions.
    handle["log_file"].close()
    trial = handle["trial"]
    result_path = handle["trial_dir"] / "result.json"
    stage = {"stage_index": stage_index, "timesteps": STAGE_TIMESTEPS[stage_index]}
    if handle["process"].returncode != 0 or not result_path.exists():
        stage["status"] = "failed"
        trial["status"] = "failed"
    else:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        stage.update({"status": "done", "score": result["score"], "last_score": result["last_score"]})
        trial["status"] = f"done_stage_{stage_index}"
        trial["model_path"] = result["model_path"]
        trial["best_model_path"] = result["best_model_path"]
        if result["score"] is not None and (state["best"]["score"] is None or result["score"] > state["best"]["score"]):
            state["best"] = {
                "trial_id": trial["trial_id"],
                "score": result["score"],
                "params": trial["params"],
                "model_path": result["best_model_path"],
            }
            (output_root / "best_result.json").write_text(json.dumps(state["best"], indent=2), encoding="utf-8")
            if Path(result["best_model_path"]).exists():
                shutil.copy2(result["best_model_path"], output_root / "best_model.zip")
    trial["stages"].append(stage)
    save_state(state, state_path)


def run_manager(args: argparse.Namespace) -> None:
    # Coordinate search stages, worker launches, and promotions.
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    state, state_path = load_or_create_state(args, output_root)

    for stage_index in range(len(STAGE_TIMESTEPS)):
        if stage_index == 0:
            queue = [trial for trial in state["trials"] if not stage_result(trial, stage_index)]
        else:
            queue = [trial for trial in state["trials"] if trial["status"] == f"promoted_stage_{stage_index - 1}"]

        running = {}
        available_gpus = gpu_pool(args)
        if not available_gpus:
            raise RuntimeError("No GPU slots available. Request at least one GPU or pass --gpus > 0.")

        while queue or running:
            while queue and available_gpus:
                trial = queue.pop(0)
                gpu_id = available_gpus.pop(0)
                trial["status"] = f"running_stage_{stage_index}"
                save_state(state, state_path)
                running[gpu_id] = launch_trial(trial, output_root, stage_index, gpu_id)

            if not running:
                break

            time.sleep(args.poll_seconds)
            for gpu_id, handle in list(running.items()):
                if handle["process"].poll() is None:
                    continue
                collect_trial(handle, state, state_path, stage_index, output_root)
                available_gpus.append(gpu_id)
                del running[gpu_id]

        completed = [trial for trial in state["trials"] if stage_result(trial, stage_index)]
        ranked = sorted(
            completed,
            key=lambda trial: stage_result(trial, stage_index).get("score") if stage_result(trial, stage_index).get("score") is not None else float("-inf"),
            reverse=True,
        )
        if stage_index == len(STAGE_TIMESTEPS) - 1:
            break
        keep = max(1, len(ranked) // 2)
        survivors = {trial["trial_id"] for trial in ranked[:keep]}
        for trial in ranked:
            trial["status"] = f"promoted_stage_{stage_index}" if trial["trial_id"] in survivors else f"pruned_stage_{stage_index}"
        save_state(state, state_path)


def main() -> None:
    # Run the script entry point.
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_manager(args)


if __name__ == "__main__":
    main()
