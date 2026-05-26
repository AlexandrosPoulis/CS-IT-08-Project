import gymnasium as gym
import numpy as np
import os
import json
import glob
import re
import imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecEnvWrapper, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from typing import Callable
from cprocgen import CProcgenEnv
import argparse
import signal
import sys
import shutil
import torch
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

out_dir = os.environ.get("TMPDIR", "/tmp")
log_path = os.path.join(out_dir, "logs.txt")

# for context logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from procgen import env

# Prevent the signal handler from running twice
is_saving = False

coinrun_win_reward = 10.0  
eval_start_level = 10000
eval_num_levels = 0 

# Wrapper (adapted for CProcgen)
class CProcgenRGBWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(15)

    def reset(self):
        obs = self.venv.reset()
        return self._get_rgb(obs)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._get_rgb(obs), rewards, dones, infos

    def _get_rgb(self, obs):
        if isinstance(obs, dict) and "rgb" in obs:
            return obs["rgb"]
        return obs  # fallback


class VecCutoutWrapper(VecEnvWrapper):
    def __init__(self, venv, n_holes=4, max_h_size=12, max_w_size=12):
        super().__init__(venv)
        self.n_holes = n_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size

    def reset(self):
        obs = self.venv.reset()
        return self._apply_cutout(obs)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._apply_cutout(obs), rewards, dones, infos

    def _apply_cutout(self, obs):
        obs = np.asarray(obs)
        if obs.ndim == 3:
            return self._cutout_single(obs)
        if obs.ndim == 4:
            return np.stack([self._cutout_single(frame) for frame in obs], axis=0)
        return obs

    def _cutout_single(self, obs):
        obs = obs.copy()
        height, width, _ = obs.shape
        for _ in range(self.n_holes):
            hole_h = np.random.randint(4, self.max_h_size + 1)
            hole_w = np.random.randint(4, self.max_w_size + 1)
            y = np.random.randint(0, max(1, height - hole_h + 1))
            x = np.random.randint(0, max(1, width - hole_w + 1))
            color = np.random.randint(0, 256, size=(3,), dtype=obs.dtype)
            obs[y : y + hole_h, x : x + hole_w] = color
        return obs

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def get_run_paths(base_name: str, folder: str):
    os.makedirs(folder, exist_ok=True)
    version = 0
    while True:
        run_name = f"{base_name}_v{version}"
        run_dir = os.path.join(folder, run_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            return run_name, run_dir, os.path.join(run_dir, f"{run_name}.zip")
        version += 1

def get_latest_existing_run_paths(base_name: str, folder: str):
    os.makedirs(folder, exist_ok=True)
    pattern = os.path.join(folder, f"{base_name}_v*")
    candidates = [path for path in glob.glob(pattern) if os.path.isdir(path)]

    if not candidates:
        return None

    def extract_version(path):
        match = re.search(r"_v(\d+)$", os.path.basename(path))
        return int(match.group(1)) if match else -1

    candidates.sort(key=extract_version)
    run_dir = candidates[-1]
    run_name = os.path.basename(run_dir)
    return run_name, run_dir, os.path.join(run_dir, f"{run_name}.zip")

def load_context(args):
    if args.context_options is None or args.context_options.strip() == "":
        return []

    try:
        ctx = json.loads(args.context_options)

        # enforce correct format
        if isinstance(ctx, dict):
            print("WARNING: context_options was a dict, converting to list")
            ctx = [ctx]

        if not isinstance(ctx, list):
            raise ValueError("context_options must be a list of dicts")

        for i, c in enumerate(ctx):
            if not isinstance(c, dict):
                raise ValueError(f"context_options[{i}] is not a dict")

        return ctx

    except Exception as e:
        print("Failed to parse context_options")
        print("Raw input:", args.context_options)
        raise e

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on C-Procgen")

    # Env params
    parser.add_argument("--env_name", type=str, default="coinrun")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--distribution_mode", type=str, default="hard")
    parser.add_argument("--start_level", type=int, default=0)
    parser.add_argument("--num_levels", type=int, default=500)
    parser.add_argument("--rand_seed", type=int, default=42)
    parser.add_argument("--frame_stack", type=int, default=4)

    # Data augmentation
    parser.add_argument("--cutout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cutout_n_holes", type=int, default=4)
    parser.add_argument("--cutout_max_h_size", type=int, default=12)
    parser.add_argument("--cutout_max_w_size", type=int, default=12)

    # Contexts (JSON string)
    parser.add_argument(
        "--context_options",
        type=str,
        default='[{"visibility": 13, "allow_monsters": true}]',
        help="JSON list of context dicts"
    )

    # Training params
    parser.add_argument("--policy_type", type=str, default="CnnPolicy")
    parser.add_argument("--total_timesteps", type=int, default=256_000_000)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--n_steps", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="tensorboard_logs/cprocgen/data_aug")
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()

class ContextEpisodeCallback(BaseCallback):
    def __init__(self, coinrun_win_reward=10.0, verbose=0):
        super().__init__(verbose)
        self.coinrun_win_reward = coinrun_win_reward
        self.data = []

    @staticmethod
    def unwrap_env(env):
        while hasattr(env, "venv"):
            env = env.venv
        return env

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None:
            return True

        base_env = self.unwrap_env(self.training_env)
        contexts = base_env.get_context()

        for i, info in enumerate(infos):
            ep = info.get("episode")
            if ep is None:
                continue

            reward = ep["r"]
            ctx = contexts[i]

            self.data.append({
                "difficulty": ctx["difficulty"],
                "num_sections": ctx["num_sections"],
                "reward": reward,
                "success": reward >= self.coinrun_win_reward
            })

        return True

    def _on_training_end(self) -> None:
        # get SB3 logger folder
        log_dir = self.logger.dir
        if log_dir is None:
            print("[ContextEpisodeCallback] No logger directory found.")
            return

        path = os.path.join(log_dir, "episode_log.csv")

        import pandas as pd
        df = pd.DataFrame(self.data)
        df.to_csv(path, index=False)

        if self.verbose:
            print(f"[ContextEpisodeCallback] Saved episode log to {path}")

class OnlineWinRateCallback(BaseCallback):
    def __init__(self, win_threshold=10.0, ema_alpha=0.9, verbose=1, state_path=None):
        super().__init__(verbose)
        self.win_threshold = win_threshold
        self.ema_alpha = ema_alpha
        self.state_path = state_path

        self.total_episodes = 0
        self.total_wins = 0
        self.win_rate_ema = 0.0

    def load_state(self) -> None:
        if self.state_path is None or not os.path.exists(self.state_path):
            return

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.total_episodes = int(data.get("total_episodes", 0))
            self.total_wins = int(data.get("total_wins", 0))
            self.win_rate_ema = float(data.get("win_rate_ema", 0.0))
        except Exception:
            if self.verbose:
                print("[OnlineWinRateCallback] Failed to load win rate state.")

    def save_state(self) -> None:
        if self.state_path is None:
            return

        data = {
            "total_episodes": self.total_episodes,
            "total_wins": self.total_wins,
            "win_rate_ema": self.win_rate_ema,
        }
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            if self.verbose:
                print("[OnlineWinRateCallback] Failed to save win rate state.")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")

        if infos is None:
            return True
        
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                reward = ep["r"]

                self.total_episodes += 1
                if reward >= self.win_threshold:
                    self.total_wins += 1

                win_rate = self.total_wins / max(1, self.total_episodes)

                # EMA smoothing
                self.win_rate_ema = (
                    self.ema_alpha * self.win_rate_ema
                    + (1 - self.ema_alpha) * win_rate
                )

        # Log every step
        self.logger.record("train/win_rate", self.total_wins / max(1, self.total_episodes))
        self.logger.record("train/win_rate_ema", self.win_rate_ema)

        return True

class FailureModeCallback(BaseCallback):
    def __init__(self, log_dir, verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.state_path = os.path.join(self.log_dir, "death_stats.json")

        # Matches procgen/src/game.h DeathType enum.
        self.death_type_map = {
            1: "saw",
            2: "enemy",
            3: "lava",
            4: "unknown",
            5: "timeout",
        }

        self.stats = {
            "saw": 0,
            "timeout": 0,
            "lava": 0,
            "enemy": 0,
            "unknown": 0,
            "total_deaths": 0,
            "total_episodes": 0,
            "total_wins": 0,
        }

        os.makedirs(log_dir, exist_ok=True)

    def load_state(self) -> None:
        if not os.path.exists(self.state_path):
            return

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            if self.verbose:
                print("[FailureModeCallback] Failed to load death stats state.")
            return

        for key in self.stats:
            if key in data and isinstance(data[key], int):
                self.stats[key] = data[key]

    def save_state(self) -> None:
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")

        if infos is None or dones is None:
            return True

        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                ep = info.get("episode")

                if ep is None:
                    continue

                self.stats["total_episodes"] += 1
                reward = ep["r"]

                prev_level_complete = info.get("prev_level_complete")
                if prev_level_complete is not None:
                    try:
                        is_win = bool(int(np.asarray(prev_level_complete).item()))
                    except Exception:
                        is_win = False
                else:
                    is_win = reward >= coinrun_win_reward

                # Count wins
                if is_win:
                    self.stats["total_wins"] += 1
                else:
                    # Only count deaths
                    self.stats["total_deaths"] += 1

                    death_type = self.decode_death_type(info)
                    self.stats[death_type] += 1

        # log continuously
        self.logger.record("failure/saw", self.stats["saw"])
        self.logger.record("failure/timeout", self.stats["timeout"])
        self.logger.record("failure/lava", self.stats["lava"])
        self.logger.record("failure/enemy", self.stats["enemy"])
        self.logger.record("failure/unknown", self.stats["unknown"])
        self.logger.record("failure/total", self.stats["total_deaths"])
        self.logger.record("failure/episodes", self.stats["total_episodes"])

        return True

    def _on_training_end(self) -> None:
        self.save_state()
        self.log_death_statistics()

    def decode_death_type(self, info):
        raw = info.get("death_type", 4)

        try:
            code = int(np.asarray(raw).item())
        except Exception:
            code = 4

        return self.death_type_map.get(code, "unknown")

    def log_death_statistics(self) -> None:
        for key in ("saw", "enemy", "lava", "timeout", "unknown"):
            self.logger.record(f"failure/{key}", self.stats[key])

        self.logger.record("failure/total", self.stats["total_deaths"])
        self.logger.record("failure/episodes", self.stats["total_episodes"])
        self.logger.dump(self.num_timesteps)

        # Calculate win rate
        win_rate = self.stats["total_wins"] / max(1, self.stats["total_episodes"])
        
        # Calculate percentages for each death type
        total_deaths = self.stats["total_deaths"]
        death_percentages = {}
        for key in ("saw", "enemy", "lava", "timeout", "unknown"):
            count = self.stats[key]
            percentage = (count / max(1, total_deaths)) * 100
            death_percentages[key] = percentage

        log_path = os.path.join(self.log_dir, "death_stat.log")
        lines = [
            f"num_timesteps: {self.num_timesteps}",
            f"rating: {win_rate:.4f}",
            f"total_wins: {self.stats['total_wins']}",
            f"total_deaths: {self.stats['total_deaths']}",
            f"total_episodes: {self.stats['total_episodes']}",
            f"",
            f"Death type breakdown:",
        ]
        for key in ("saw", "enemy", "lava", "timeout", "unknown"):
            count = self.stats[key]
            percentage = death_percentages[key]
            lines.append(f"  {key}: {count} ({percentage:.2f}%)")

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        if self.verbose:
            print("\nFinal death statistics:")
            print(f"  Rating (win rate): {win_rate:.4f}")
            print(f"  Total wins: {self.stats['total_wins']}")
            print(f"  Total deaths: {self.stats['total_deaths']}")
            print(f"  Total episodes: {self.stats['total_episodes']}")
            print(f"\nDeath type breakdown:")
            for key in ("saw", "enemy", "lava", "timeout", "unknown"):
                count = self.stats[key]
                percentage = death_percentages[key]
                print(f"    {key}: {count} ({percentage:.2f}%)")

def main():
    args = parse_args()
    hyperparams = vars(args)

    # Parse context JSON string to Python object
    print(args.context_options)
    context_options = load_context(args)
    print("Initializing C-Procgen environments...")

    train_env = CProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env_name,
        start_level=args.start_level,
        num_levels=args.num_levels,
        context_options=context_options,
        distribution_mode=args.distribution_mode,
        rand_seed=args.rand_seed,
    )

    train_env = CProcgenRGBWrapper(train_env)
    if args.cutout:
        train_env = VecCutoutWrapper(
            train_env,
            n_holes=args.cutout_n_holes,
            max_h_size=args.cutout_max_h_size,
            max_w_size=args.cutout_max_w_size,
        )
    train_env = VecMonitor(train_env)
    if args.frame_stack > 0:
        train_env = VecFrameStack(train_env, n_stack=args.frame_stack)
        

    eval_env = CProcgenEnv(
        num_envs=min(16, args.num_envs),  # smaller eval env
        env_name=args.env_name,
        start_level=eval_start_level,
        num_levels=eval_num_levels,
        distribution_mode=args.distribution_mode,
        context_options=context_options,
    )

    eval_env = CProcgenRGBWrapper(eval_env)
    eval_env = VecMonitor(eval_env)

    if args.frame_stack > 0:
        eval_env = VecFrameStack(eval_env, n_stack=args.frame_stack)

    # Logging setup
    run_tag = "_".join([
        args.env_name,
        f"{args.batch_size}",
        f"{args.frame_stack}",
        f"levels{args.num_levels}",
        f"seed{args.rand_seed}",
    ])
    timesteps_m = args.total_timesteps // 1_000_000
    base_model_name = f"{args.policy_type}_{timesteps_m}M_{run_tag}"

    model_root = os.path.join(
        args.save_dir,
        f"levels{args.num_levels}",
    )
    if args.resume:
        resumed_run = get_latest_existing_run_paths(base_model_name, model_root)
        if resumed_run is not None:
            run_name, run_dir, model_path = resumed_run
        else:
            run_name, run_dir, model_path = get_run_paths(base_model_name, model_root)
    else:
        run_name, run_dir, model_path = get_run_paths(base_model_name, model_root)

    print("Run folder:", run_dir)
    print("Model will be saved as:", model_path)
    
    tensorboard_dir = run_dir
    log_data = {
        "hyperparameters": hyperparams,
        "context_options": context_options,   # log contexts
        "run_name": run_name,
        "model_path": model_path,
        "tensorboard_dir": tensorboard_dir,
    }

    with open(os.path.join(tensorboard_dir, "training_config.json"), "w") as f:
        json.dump(log_data, f, indent=4)
    
    # We'll store checkpoints in the tensorboard run folder so checkpoints and logs live together.
    checkpoint_folder = run_dir
    os.makedirs(checkpoint_folder, exist_ok=True)

    # Default paths
    checkpoint_path = os.path.join(checkpoint_folder, f"{run_name}_latest.zip")
    final_model_path = os.path.join(checkpoint_folder, f"{run_name}_final.zip")
   
    # Create or resume model
    # Only resume from interrupt checkpoints to keep rollout-boundary alignment.
    interrupt_files = sorted(
        glob.glob(os.path.join(checkpoint_folder, f"{run_name}_interrupt_*_steps.zip"))
    )

    if args.resume and len(interrupt_files) > 0:
        # pick the interrupt checkpoint with the largest step count
        def extract_steps(p):
            m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(p))
            return int(m.group(1)) if m else -1

        interrupt_files.sort(key=extract_steps)
        checkpoint_path = interrupt_files[-1]

        print(f"Resuming from interrupt checkpoint: {checkpoint_path}")

        model = PPO.load(checkpoint_path, env=train_env, device=args.device)
        completed_timesteps = model.num_timesteps
        print(f"Completed timesteps: {completed_timesteps:,}")

    else:
        print("Creating new PPO model")

        model = PPO(
            args.policy_type,
            train_env,
            learning_rate=linear_schedule(args.learning_rate),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            vf_coef=args.vf_coef,
            verbose=1,
            device=args.device,
        )
        completed_timesteps = 0

    print(model.policy)
    print(model.policy.features_extractor)
    print(model.policy.mlp_extractor)
    
    # Compute remaining timesteps
    target_timesteps = args.total_timesteps
    remaining_timesteps = target_timesteps - completed_timesteps

    print(f"Target timesteps:    {target_timesteps:,}")
    print(f"Completed timesteps: {completed_timesteps:,}")
    print(f"Remaining timesteps: {remaining_timesteps:,}")

    if remaining_timesteps <= 0:
        print("Training already completed. Skipping training.")

        # Save final model to a consistent filename
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")

        print("Done.")
        return

    # Callbacks

    context_callback = ContextEpisodeCallback(
        coinrun_win_reward=coinrun_win_reward,
        verbose=1
    )

    win_callback = OnlineWinRateCallback(
        win_threshold=coinrun_win_reward,
        ema_alpha=0.95,
        verbose=1,
        state_path=os.path.join(run_dir, "win_rate_state.json"),
    )

    failure_callback = FailureModeCallback(
        log_dir=run_dir,
        verbose=1,
    )

    if args.resume:
        win_callback.load_state()
        failure_callback.load_state()
        

    # Prevent the signal handler from running twice
    is_saving = False

    def save_and_exit(signum, frame):
        nonlocal is_saving
        if is_saving:
            return
        is_saving = True

        print(f"\nReceived signal {signum}. Stopping training and saving checkpoint...", flush=True)

        rollout_size = args.n_steps * args.num_envs
        last_boundary = (model.num_timesteps // rollout_size) * rollout_size

        if last_boundary < model.num_timesteps:
            print(
                f"Rounding down timesteps from {model.num_timesteps:,} "
                f"to {last_boundary:,} to align with rollout boundary ({rollout_size}).",
                flush=True,
            )

        # Stop training cleanly at boundary
        model._total_timesteps = last_boundary

        interrupt_path = os.path.join(
            run_dir,
            f"{run_name}_interrupt_{last_boundary}_steps.zip"
        )

        tmp_path = f"/tmp/{run_name}_interrupt.zip"

        try:
            
            # 1. Move model to CPU (avoids slow GPU sync issues)
            if hasattr(model, "policy"):
                model.policy.to("cpu")

            # 2. Save locally first (FAST, avoids NFS death)
            model.save(tmp_path)

            # 3. Copy to final destination (persistent storage)
            shutil.copy2(tmp_path, interrupt_path)
            print(f"Interrupt checkpoint saved to {interrupt_path}", flush=True)

            # 4. Save training statistics
            win_callback.save_state()
            failure_callback.save_state()
            failure_callback.log_death_statistics()

            # 5. Flush TensorBoard safely
            if hasattr(model, "logger") and model.logger is not None:
                try:
                    model.logger.dump(last_boundary)
                except Exception as e:
                    print(f"Warning: TensorBoard flush failed: {e}", flush=True)

            # 6. Save final model too (same checkpoint)
            final_tmp = f"/tmp/{run_name}_final.zip"
            model.save(final_tmp)
            shutil.copy2(final_tmp, final_model_path)

            print(f"Final model saved to {final_model_path}", flush=True)

        except Exception as e:
            print(f"ERROR DURING INTERRUPT SAVE: {e}", flush=True)

        finally:
            print("Training complete (forced shutdown).", flush=True)
            os._exit(0)

    # Handle scheduler termination and Ctrl+C
    signal.signal(signal.SIGTERM, save_and_exit)  # scheduler termination
    signal.signal(signal.SIGINT, save_and_exit)   # Ctrl+C

    model.set_logger(configure(run_dir, ["stdout", "tensorboard"]))

    # Train
    print(f"Starting training for {remaining_timesteps:,} additional steps...")
    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            progress_bar=True,
            reset_num_timesteps=not args.resume,
            callback=[
                win_callback,
                failure_callback,
                context_callback]
        )
    # NORMAL completion path
    finally:
        print("Training finished normally. Saving final model...", flush=True)

        final_path = os.path.join(run_dir, f"{run_name}_final.zip")
        tmp_path = f"/tmp/{run_name}_final.zip"

        try:
            if hasattr(model, "policy"):
                model.policy.to("cpu")

            model.save(tmp_path)
            shutil.copy2(tmp_path, final_path)

            print(f"Final model saved to {final_path}", flush=True)

        except Exception as e:
            print(f"ERROR DURING FINAL SAVE: {e}", flush=True)

        # Always cleanup
        for env in (train_env, eval_env):
            try:
                env.close()
            except Exception as e:
                print(f"Env close warning: {e}", flush=True)

        try:
            if hasattr(model, "logger") and model.logger is not None:
                model.logger.dump(model.num_timesteps)
                if hasattr(model.logger, "close"):
                    model.logger.close()
        except Exception as e:
            print(f"Logger final flush failed: {e}", flush=True)

if __name__ == "__main__":
    main()