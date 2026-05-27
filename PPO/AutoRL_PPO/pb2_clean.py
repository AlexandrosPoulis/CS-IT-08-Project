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


DEFAULT_TOTAL_TIMESTEPS = 50_000_000
DEFAULT_SEGMENT_TIMESTEPS = 2_000_000
DEFAULT_POPULATION_SIZE = 4
TRAIN_NUM_ENVS = 64
EVAL_NUM_ENVS = 64
EVAL_EPISODES = 256

PARAM_SPACE = {
    "learning_rate": {"type": "log_float", "min": 1e-4, "max": 8e-4},
    "num_steps": {"type": "choice", "values": [128, 256]},
    "num_minibatches": {"type": "choice", "values": [4, 8, 16]},
    "update_epochs": {"type": "choice", "values": [3, 6, 10]},
    "gamma": {"type": "choice", "values": [0.99, 0.995, 0.999]},
    "gae_lambda": {"type": "choice", "values": [0.95, 0.97, 0.99]},
    "ent_coef": {"type": "log_float", "min": 1e-3, "max": 2e-2},
    "clip_coef": {"type": "choice", "values": [0.1, 0.2, 0.3]},
    "vf_coef": {"type": "choice", "values": [0.5, 0.75, 1.0]},
    "max_grad_norm": {"type": "choice", "values": [0.5, 1.0]},
}


def parse_args() -> argparse.Namespace:
    # Parse command-line options for this script.
    parser = argparse.ArgumentParser(description="PB2-style population search wrapper for clean_ppo_m.py")
    parser.add_argument("--output_root", default="pb2_runs")
    parser.add_argument("--base_module", default="clean_ppo_m")
    parser.add_argument("--env_name", default="coinrun")
    parser.add_argument("--distribution_mode", "--distribution-mode", choices=("easy", "hard"), default="easy")
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
        "distribution_mode": args.distribution_mode,
        "algorithm": "clean_ppo",
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
        shutil.copy2(result["model_path"], output_root / "best_model.pt")


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
    distribution_mode: str,
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
        "--distribution_mode",
        distribution_mode,
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
                    args.distribution_mode,
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


def make_procgen_env(env_name: str, distribution_mode: str, num_envs: int, gamma: float, normalize_reward: bool):
    # Build a Procgen vector environment for training or evaluation.
    import gym
    from procgen import ProcgenEnv

    envs = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        num_levels=0,
        start_level=0 if normalize_reward else 50_000,
        distribution_mode=distribution_mode,
    )
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    if normalize_reward:
        envs = gym.wrappers.NormalizeReward(envs, gamma=gamma)
        envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    return envs


def find_normalize_reward(envs):
    # Find the reward-normalization wrapper in a vector environment.
    current = envs
    while current is not None:
        if current.__class__.__name__ == "NormalizeReward":
            return current
        current = getattr(current, "env", None)
    return None


def load_checkpoint(agent, optimizer, train_env, checkpoint_path: str, device) -> int:
    # Restore model, optimizer, environment stats, and step count.
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    if not checkpoint or not checkpoint.exists():
        return 0

    import torch

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(payload["model_state_dict"])
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    reward_state = payload.get("reward_normalization")
    reward_wrapper = find_normalize_reward(train_env)
    if reward_state and reward_wrapper is not None:
        reward_wrapper.return_rms.mean = reward_state["mean"]
        reward_wrapper.return_rms.var = reward_state["var"]
        reward_wrapper.return_rms.count = reward_state["count"]
        reward_wrapper.returns = reward_state["returns"]
    return int(payload.get("global_step", 0))


def save_checkpoint(agent, optimizer, train_env, train_args, model_path: Path, global_step: int, params: dict) -> None:
    # Save model weights, optimizer state, environment stats, and params.
    reward_wrapper = find_normalize_reward(train_env)
    reward_state = None
    if reward_wrapper is not None:
        reward_state = {
            "mean": reward_wrapper.return_rms.mean,
            "var": reward_wrapper.return_rms.var,
            "count": reward_wrapper.return_rms.count,
            "returns": reward_wrapper.returns,
        }

    import torch

    torch.save(
        {
            "args": vars(train_args),
            "params": params,
            "global_step": global_step,
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "reward_normalization": reward_state,
        },
        model_path,
    )


def make_train_args(base, args: argparse.Namespace, params: dict):
    # Merge PB2 parameters into Clean PPO training arguments.
    train_args = base.Args()
    train_args.env_id = args.env_name
    train_args.distribution_mode = args.distribution_mode
    train_args.total_timesteps = args.timesteps
    train_args.num_envs = TRAIN_NUM_ENVS
    train_args.learning_rate = params["learning_rate"]
    train_args.num_steps = params["num_steps"]
    train_args.num_minibatches = params["num_minibatches"]
    train_args.update_epochs = params["update_epochs"]
    train_args.gamma = params["gamma"]
    train_args.gae_lambda = params["gae_lambda"]
    train_args.ent_coef = params["ent_coef"]
    train_args.clip_coef = params["clip_coef"]
    train_args.vf_coef = params["vf_coef"]
    train_args.max_grad_norm = params["max_grad_norm"]
    train_args.batch_size = int(train_args.num_envs * train_args.num_steps)
    train_args.minibatch_size = int(train_args.batch_size // train_args.num_minibatches)
    train_args.num_iterations = train_args.total_timesteps // train_args.batch_size
    return train_args


def evaluate_agent(agent, eval_env, device, eval_episodes: int, win_reward: float) -> dict:
    # Evaluate a Clean PPO agent and summarize episode metrics.
    import torch
    from torch.distributions.categorical import Categorical

    obs = eval_env.reset()
    ep_rewards = np.zeros(EVAL_NUM_ENVS, dtype=np.float64)
    ep_lengths = np.zeros(EVAL_NUM_ENVS, dtype=np.int64)
    completed = 0
    wins = 0
    rewards: list[float] = []
    lengths: list[float] = []

    while completed < eval_episodes:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            hidden = agent.network(obs_tensor.permute((0, 3, 1, 2)) / 255.0)
            actions = Categorical(logits=agent.actor(hidden)).sample().cpu().numpy()

        obs, step_rewards, dones, _infos = eval_env.step(actions)
        ep_rewards += step_rewards
        ep_lengths += 1
        for idx, done in enumerate(dones):
            if not done:
                continue
            reward = float(ep_rewards[idx])
            length = float(ep_lengths[idx])
            rewards.append(reward)
            lengths.append(length)
            wins += int(reward >= win_reward)
            completed += 1
            ep_rewards[idx] = 0.0
            ep_lengths[idx] = 0
            if completed >= eval_episodes:
                break

    return {
        "episodes": eval_episodes,
        "wins": wins,
        "win_rate": 100.0 * wins / eval_episodes,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def run_worker(args: argparse.Namespace) -> None:
    # Train and evaluate one worker segment, then write its result.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    import importlib

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter

    base = importlib.import_module(args.base_module)

    member_dir = Path(args.member_dir)
    round_dir = member_dir / f"round_{args.round_index:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    params = json.loads(args.params_json)
    train_args = make_train_args(base, args, params)

    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    torch.backends.cudnn.deterministic = train_args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and train_args.cuda else "cpu")
    train_env = make_procgen_env(args.env_name, args.distribution_mode, TRAIN_NUM_ENVS, train_args.gamma, normalize_reward=True)
    eval_env = make_procgen_env(args.env_name, args.distribution_mode, EVAL_NUM_ENVS, train_args.gamma, normalize_reward=False)
    agent = base.Agent(train_env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=train_args.learning_rate, eps=1e-5)
    global_step = load_checkpoint(agent, optimizer, train_env, args.checkpoint_path, device)
    for group in optimizer.param_groups:
        group["lr"] = train_args.learning_rate

    writer = SummaryWriter(str(round_dir / "tensorboard"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(train_args).items()])),
    )

    obs = torch.zeros((train_args.num_steps, train_args.num_envs) + train_env.single_observation_space.shape).to(device)
    actions = torch.zeros((train_args.num_steps, train_args.num_envs) + train_env.single_action_space.shape).to(device)
    logprobs = torch.zeros((train_args.num_steps, train_args.num_envs)).to(device)
    rewards = torch.zeros((train_args.num_steps, train_args.num_envs)).to(device)
    dones = torch.zeros((train_args.num_steps, train_args.num_envs)).to(device)
    values = torch.zeros((train_args.num_steps, train_args.num_envs)).to(device)

    start_time = time.time()
    next_obs = torch.Tensor(train_env.reset()).to(device)
    next_done = torch.zeros(train_args.num_envs).to(device)

    try:
        for iteration in range(1, train_args.num_iterations + 1):
            if train_args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / train_args.num_iterations
                lrnow = frac * train_args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, train_args.num_steps):
                global_step += train_args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, next_done, info = train_env.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                for item in info:
                    if "episode" in item.keys():
                        print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        break

            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(train_args.num_steps)):
                    if t == train_args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + train_args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + train_args.gamma * train_args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            b_obs = obs.reshape((-1,) + train_env.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + train_env.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(train_args.batch_size)
            clipfracs = []
            for _epoch in range(train_args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, train_args.batch_size, train_args.minibatch_size):
                    end = start + train_args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > train_args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if train_args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - train_args.clip_coef, 1 + train_args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if train_args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -train_args.clip_coef,
                            train_args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - train_args.ent_coef * entropy_loss + v_loss * train_args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), train_args.max_grad_norm)
                    optimizer.step()

                if train_args.target_kl is not None and approx_kl > train_args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / max(time.time() - start_time, 1e-9)), global_step)

        model_path = round_dir / "model.pt"
        save_checkpoint(agent, optimizer, train_env, train_args, model_path, global_step, params)
        metrics = evaluate_agent(agent, eval_env, device, EVAL_EPISODES, 10.0)
    finally:
        train_env.close()
        eval_env.close()
        writer.close()

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
        "model_path": str(model_path.resolve()),
        "params": params,
        "checkpoint_path": args.checkpoint_path,
        "distribution_mode": args.distribution_mode,
        "global_step": global_step,
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
