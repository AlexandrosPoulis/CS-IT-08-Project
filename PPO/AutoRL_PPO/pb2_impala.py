import argparse
import importlib
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


DEFAULT_TOTAL_TIMESTEPS = 50_000_000
DEFAULT_SEGMENT_TIMESTEPS = 2_000_000
DEFAULT_POPULATION_SIZE = 4
TRAIN_NUM_ENVS = 128
EVAL_NUM_ENVS = 128
EVAL_EPISODES = 256

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
    parser = argparse.ArgumentParser(description="PB2-style population search wrapper for main_impala.py")
    parser.add_argument("--output_root", default="pb2_runs")
    parser.add_argument("--base_module", default="main_impala")
    parser.add_argument("--env_name", default="coinrun")
    parser.add_argument("--population_size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--parallel_agents", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--gpus", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--master_seed", type=int, default=1234)
    parser.add_argument("--poll_seconds", type=int, default=5)
    parser.add_argument("--total_timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--segment_timesteps", type=int, default=DEFAULT_SEGMENT_TIMESTEPS)
    parser.add_argument("--replace_fraction", type=float, default=0.25)
    parser.add_argument("--max_worker_failures", type=int, default=2)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--member_id", default="")
    parser.add_argument("--member_dir", default="")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--round_index", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=0)
    parser.add_argument("--params_json", default="")
    parser.add_argument("--checkpoint_path", default="")
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


def mutate_params(base_params: dict, rng: random.Random) -> dict:
    # Perturb a parent hyperparameter configuration for exploration.
    params = dict(base_params)
    for name, spec in PARAM_SPACE.items():
        if spec["type"] == "choice":
            if rng.random() < 0.6:
                params[name] = rng.choice(spec["values"])
        else:
            value = math.log10(params[name])
            spread = 0.25 * (math.log10(spec["max"]) - math.log10(spec["min"]))
            new_value = min(math.log10(spec["max"]), max(math.log10(spec["min"]), value + rng.uniform(-spread, spread)))
            params[name] = 10 ** new_value
    return params


def encode_params(progress: float, params: dict) -> np.ndarray:
    # Convert progress and hyperparameters into a PB2 feature vector.
    values = [progress]
    for name, spec in PARAM_SPACE.items():
        if spec["type"] == "choice":
            options = spec["values"]
            if len(options) == 1:
                values.append(0.0)
            else:
                values.append(options.index(params[name]) / (len(options) - 1))
        else:
            low = math.log10(spec["min"])
            high = math.log10(spec["max"])
            values.append((math.log10(params[name]) - low) / (high - low))
    return np.asarray(values, dtype=np.float64)


def rbf_kernel(x: np.ndarray, y: np.ndarray, lengthscale: float = 0.35) -> np.ndarray:
    # Compute the RBF kernel used by PB2's GP surrogate.
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    diff = x[:, None, :] - y[None, :, :]
    return np.exp(-0.5 * np.sum(diff * diff, axis=2) / (lengthscale * lengthscale))


def suggest_pb2_params(history: list[dict], base_params: dict, progress: float, rng: random.Random) -> dict:
    # Suggest the next PB2 hyperparameters from observed history.
    if len(history) < 4:
        return mutate_params(base_params, rng)

    x_hist = np.asarray([encode_params(item["progress"], item["params"]) for item in history], dtype=np.float64)
    y_hist = np.asarray([item["score"] for item in history], dtype=np.float64)
    y_mean = float(np.mean(y_hist))
    y_std = float(np.std(y_hist)) + 1e-8
    y_norm = (y_hist - y_mean) / y_std

    kernel = rbf_kernel(x_hist, x_hist) + 1e-3 * np.eye(len(x_hist))
    alpha = np.linalg.solve(kernel, y_norm)

    candidates = [mutate_params(base_params, rng) for _ in range(48)]
    candidates.extend(sample_params(rng) for _ in range(16))

    best_params = candidates[0]
    best_value = float("-inf")
    for params in candidates:
        x = encode_params(progress, params)[None, :]
        k = rbf_kernel(x_hist, x).reshape(-1)
        mean = float(k @ alpha)
        var = max(1e-6, 1.0 - float(k @ np.linalg.solve(kernel, k)))
        acquisition = mean + 0.6 * math.sqrt(var)
        if acquisition > best_value:
            best_value = acquisition
            best_params = params
    return best_params


def save_state(state: dict, state_path: Path) -> None:
    # Persist the search manager state to disk.
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_or_create_state(args: argparse.Namespace, output_root: Path) -> tuple[dict, Path]:
    # Resume an existing search state or initialize a new one.
    state_path = output_root / "pb2_state.json"
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8")), state_path

    rng = random.Random(args.master_seed)
    state = {
        "round_index": 0,
        "total_timesteps": args.total_timesteps,
        "segment_timesteps": args.segment_timesteps,
        "population_size": args.population_size,
        "base_module": args.base_module,
        "env_name": args.env_name,
        "algorithm": "impala",
        "failed_gpus": [],
        "history": [],
        "best": {"member_id": "", "score": None, "params": {}, "model_path": ""},
        "members": [],
    }
    for i in range(args.population_size):
        state["members"].append(
            {
                "member_id": f"member_{i:02d}",
                "params": sample_params(rng),
                "checkpoint_path": "",
                "timesteps": 0,
                "score": None,
                "avg_reward": None,
                "win_rate": None,
                "rounds": [],
            }
        )
    save_state(state, state_path)
    return state, state_path


def update_best(state: dict, member: dict, result: dict, output_root: Path) -> None:
    # Update the best checkpoint/result seen so far.
    if result["score"] is None:
        return
    if state["best"]["score"] is not None and result["score"] <= state["best"]["score"]:
        return
    state["best"] = {
        "member_id": member["member_id"],
        "score": result["score"],
        "win_rate": result["win_rate"],
        "avg_reward": result["avg_reward"],
        "params": member["params"],
        "model_path": result["model_path"],
    }
    (output_root / "best_result.json").write_text(json.dumps(state["best"], indent=2), encoding="utf-8")
    if Path(result["model_path"]).exists():
        shutil.copy2(result["model_path"], output_root / "best_model.zip")


def available_gpu_ids(args: argparse.Namespace, state: dict) -> list[int]:
    # Return GPUs not currently occupied by active workers.
    failed_gpus = {int(gpu_id) for gpu_id in state.get("failed_gpus", [])}
    return [
        gpu_id
        for gpu_id in range(min(args.parallel_agents, args.gpus))
        if gpu_id not in failed_gpus
    ]


def apply_exploitation(state: dict, args: argparse.Namespace) -> None:
    # Replace weak PB2 members with strong performers and mutated params.
    members = [member for member in state["members"] if member["score"] is not None]
    if len(members) < 2:
        return

    ranked = sorted(members, key=lambda item: item["score"], reverse=True)
    replace_count = max(1, int(args.replace_fraction * len(ranked)))
    elites = ranked[:replace_count]
    losers = ranked[-replace_count:]
    rng = random.Random(args.master_seed + state["round_index"])

    for index, loser in enumerate(losers):
        donor = elites[index % len(elites)]
        if donor["member_id"] == loser["member_id"]:
            continue
        progress = min(1.0, (loser["timesteps"] + args.segment_timesteps) / args.total_timesteps)
        loser["params"] = suggest_pb2_params(state["history"], donor["params"], progress, rng)
        loser["checkpoint_path"] = donor["checkpoint_path"]
        loser["source_member"] = donor["member_id"]


def launch_member(
    member: dict,
    output_root: Path,
    round_index: int,
    gpu_id: int,
    segment_timesteps: int,
    base_module: str,
    env_name: str,
) -> dict:
    # Start one PB2 population member as a worker subprocess.
    member_dir = output_root / member["member_id"]
    member_dir.mkdir(parents=True, exist_ok=True)
    (member_dir / "config.json").write_text(json.dumps(member["params"], indent=2), encoding="utf-8")
    log_path = member_dir / f"round_{round_index:03d}.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--output_root",
        str(output_root),
        "--base_module",
        base_module,
        "--env_name",
        env_name,
        "--member_id",
        member["member_id"],
        "--member_dir",
        str(member_dir),
        "--gpu_id",
        str(gpu_id),
        "--round_index",
        str(round_index),
        "--timesteps",
        str(segment_timesteps),
        "--params_json",
        json.dumps(member["params"]),
        "--checkpoint_path",
        member.get("checkpoint_path", ""),
    ]
    log_file = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    return {"member": member, "process": process, "log_file": log_file, "gpu_id": gpu_id, "member_dir": member_dir}


def collect_member(
    handle: dict,
    state: dict,
    state_path: Path,
    output_root: Path,
    max_worker_failures: int,
) -> None:
    # Record a finished PB2 worker result and update manager state.
    handle["log_file"].close()
    member = handle["member"]
    result_path = handle["member_dir"] / f"round_{state['round_index']:03d}" / "result.json"
    if handle["process"].returncode != 0 or not result_path.exists():
        failures = int(member.get("failures", 0)) + 1
        member["failures"] = failures
        member["last_failed_round"] = state["round_index"]
        member["last_failed_gpu"] = handle["gpu_id"]
        member["status"] = "failed" if failures < max_worker_failures else "disabled_after_failures"
        failed_gpus = {int(gpu_id) for gpu_id in state.get("failed_gpus", [])}
        failed_gpus.add(int(handle["gpu_id"]))
        state["failed_gpus"] = sorted(failed_gpus)
        save_state(state, state_path)
        return

    result = json.loads(result_path.read_text(encoding="utf-8"))
    member["status"] = f"done_round_{state['round_index']}"
    member["failures"] = 0
    member["timesteps"] += result["segment_timesteps"]
    member["score"] = result["score"]
    member["avg_reward"] = result["avg_reward"]
    member["win_rate"] = result["win_rate"]
    member["checkpoint_path"] = result["model_path"]
    member["rounds"].append(result)

    state["history"].append(
        {
            "member_id": member["member_id"],
            "round_index": state["round_index"],
            "progress": min(1.0, member["timesteps"] / state["total_timesteps"]),
            "score": result["score"],
            "params": member["params"],
        }
    )
    update_best(state, member, result, output_root)
    save_state(state, state_path)


def run_manager(args: argparse.Namespace) -> None:
    # Coordinate search stages, worker launches, and promotions.
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    state, state_path = load_or_create_state(args, output_root)
    state.setdefault("failed_gpus", [])

    while True:
        active_members = [
            member
            for member in state["members"]
            if member["timesteps"] < args.total_timesteps
            and int(member.get("failures", 0)) < args.max_worker_failures
        ]
        if not active_members:
            break

        round_index = state["round_index"]
        running = {}
        gpu_pool = available_gpu_ids(args, state)
        if not gpu_pool:
            raise RuntimeError("No GPU slots available after excluding failed GPUs.")
        queue = list(active_members)

        while queue or running:
            while queue and gpu_pool:
                member = queue.pop(0)
                gpu_id = gpu_pool.pop(0)
                segment_timesteps = min(args.segment_timesteps, args.total_timesteps - member["timesteps"])
                member["status"] = f"running_round_{round_index}"
                save_state(state, state_path)
                running[gpu_id] = launch_member(
                    member,
                    output_root,
                    round_index,
                    gpu_id,
                    segment_timesteps,
                    args.base_module,
                    args.env_name,
                )

            if not running:
                break

            time.sleep(args.poll_seconds)
            for gpu_id, handle in list(running.items()):
                if handle["process"].poll() is None:
                    continue
                collect_member(handle, state, state_path, output_root, args.max_worker_failures)
                if int(gpu_id) not in {int(item) for item in state.get("failed_gpus", [])}:
                    gpu_pool.append(gpu_id)
                del running[gpu_id]

        state["round_index"] += 1
        apply_exploitation(state, args)
        save_state(state, state_path)


def constant_schedule(value: float):
    # Return an SB3-compatible constant hyperparameter schedule.
    return lambda _: float(value)


def impala_policy_kwargs(base) -> dict:
    # Build SB3 policy kwargs for the IMPALA feature extractor.
    return {
        "features_extractor_class": base.ImpalaCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }


def evaluate_model(model, eval_env, eval_episodes: int, win_reward: float) -> dict:
    # Evaluate a model and summarize win/reward/length metrics.
    obs = eval_env.reset()
    completed = 0
    wins = 0
    rewards: list[float] = []
    lengths: list[float] = []

    while completed < eval_episodes:
        actions, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = eval_env.step(actions)
        for idx, done in enumerate(dones):
            if not done:
                continue
            episode_info = infos[idx].get("episode", {})
            reward = float(episode_info.get("r", 0.0))
            length = float(episode_info.get("l", 0.0))
            rewards.append(reward)
            lengths.append(length)
            wins += int(reward >= win_reward)
            completed += 1
            if completed >= eval_episodes:
                break

    return {
        "episodes": eval_episodes,
        "wins": wins,
        "win_rate": 100.0 * wins / eval_episodes,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def build_or_load_model(train_env, round_dir: Path, params: dict, checkpoint_path: str, base):
    # Create or resume a PPO model for one worker segment.
    from stable_baselines3 import PPO

    tensorboard_dir = round_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    policy_kwargs = impala_policy_kwargs(base)

    if checkpoint and checkpoint.exists():
        model = PPO.load(str(checkpoint), env=train_env, device="cuda", custom_objects={"policy_kwargs": policy_kwargs})
        model.tensorboard_log = str(tensorboard_dir)
    else:
        model = PPO(
            "CnnPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=params["learning_rate"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            gamma=params["gamma"],
            gae_lambda=params["gae_lambda"],
            ent_coef=params["ent_coef"],
            clip_range=params["clip_range"],
            vf_coef=params["vf_coef"],
            max_grad_norm=params["max_grad_norm"],
            device="cuda",
            tensorboard_log=str(tensorboard_dir),
        )
        return model, False

    model.learning_rate = params["learning_rate"]
    model.lr_schedule = constant_schedule(params["learning_rate"])
    model.n_steps = params["n_steps"]
    model.batch_size = params["batch_size"]
    model.n_epochs = params["n_epochs"]
    model.gamma = params["gamma"]
    model.gae_lambda = params["gae_lambda"]
    model.ent_coef = params["ent_coef"]
    model.clip_range = constant_schedule(params["clip_range"])
    model.vf_coef = params["vf_coef"]
    model.max_grad_norm = params["max_grad_norm"]
    model.rollout_buffer = model.rollout_buffer_class(
        model.n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
        **getattr(model, "rollout_buffer_kwargs", {}),
    )
    return model, True


def run_worker(args: argparse.Namespace) -> None:
    # Train and evaluate one worker segment, then write its result.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    from procgen import ProcgenEnv
    from stable_baselines3.common.vec_env import VecMonitor
    base = importlib.import_module(args.base_module)

    member_dir = Path(args.member_dir)
    round_dir = member_dir / f"round_{args.round_index:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    params = json.loads(args.params_json)
    seed = getattr(base, "global_seed", 1)

    train_env = VecMonitor(
        base.ProcgenRGBWrapper(
            ProcgenEnv(
                num_envs=TRAIN_NUM_ENVS,
                env_name=args.env_name,
                start_level=0,
                num_levels=0,
                distribution_mode="hard",
                rand_seed=seed,
            )
        )
    )
    eval_env = VecMonitor(
        base.ProcgenRGBWrapper(
            ProcgenEnv(
                num_envs=EVAL_NUM_ENVS,
                env_name=args.env_name,
                start_level=base.eval_start_level,
                num_levels=base.eval_num_levels,
                distribution_mode="hard",
                rand_seed=seed,
            )
        )
    )

    model, resumed = build_or_load_model(train_env, round_dir, params, args.checkpoint_path, base)
    try:
        model.learn(
            total_timesteps=args.timesteps,
            progress_bar=True,
            reset_num_timesteps=not resumed,
            tb_log_name=args.member_id,
        )
        model_path = round_dir / "model"
        model.save(str(model_path))
        win_reward = getattr(base, f"{args.env_name}_win_reward", getattr(base, "coinrun_win_reward", 10.0))
        metrics = evaluate_model(model, eval_env, EVAL_EPISODES, win_reward)
    finally:
        train_env.close()
        eval_env.close()

    summary = {
        "member_id": args.member_id,
        "round_index": args.round_index,
        "segment_timesteps": args.timesteps,
        "score": metrics["avg_reward"],
        "avg_reward": metrics["avg_reward"],
        "avg_length": metrics["avg_length"],
        "win_rate": metrics["win_rate"],
        "wins": metrics["wins"],
        "episodes": metrics["episodes"],
        "model_path": str((round_dir / "model.zip").resolve()),
        "params": params,
        "checkpoint_path": args.checkpoint_path,
    }
    (round_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    # Run the script entry point.
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_manager(args)


if __name__ == "__main__":
    main()
