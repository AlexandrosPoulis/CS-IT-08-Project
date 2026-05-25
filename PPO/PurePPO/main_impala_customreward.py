import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecEnvWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces
from procgen import ProcgenEnv
import numpy as np
from typing import Callable

eval_start_level = 100_000
eval_num_levels = 1_000
coinrun_win_reward = 10.0
global_seed = 1

# ── Wrappers ──────────────────────────────────────────────────────────────────

# https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
class ProcgenRGBWrapper(VecEnvWrapper):
    """Extracts the RGB observation and transposes to channels-first (C, H, W)."""
    def __init__(self, venv):
        super().__init__(venv)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(15)

    def reset(self):
        obs = self.venv.reset()
        return obs["rgb"].transpose(0, 3, 1, 2)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs["rgb"].transpose(0, 3, 1, 2), rewards, dones, infos


class RewardShapingWrapper(VecEnvWrapper):
    """Applies a per-step time penalty to encourage faster episode completion."""
    def __init__(self, venv, time_penalty: float = 0.01):
        super().__init__(venv)
        self.time_penalty = time_penalty

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        shaped_rewards = rewards.copy()
        shaped_rewards -= self.time_penalty
        return obs, shaped_rewards, dones, infos


# ── IMPALA CNN architecture ───────────────────────────────────────────────────
# https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/master/models/impala_cnn_torch.py

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        # resnet-style skip connection: input is added to the output of the conv layers
        out = nn.functional.relu(x)
        out = self.conv1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        return out + residual


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv    = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1    = ResidualBlock(out_channels)
        self.res2    = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # SB3 feeds images as (C, H, W) channels-first
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            ImpalaBlock(n_input_channels, 16),
            ImpalaBlock(16, 32),
            ImpalaBlock(32, 32),
            nn.ReLU(),
            nn.Flatten(),
        )

        # infer the flattened size with a dummy forward pass so we don't have to
        # manually calculate the output shape
        # https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html
        with th.no_grad():
            dummy = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # normalize pixels [0, 255] → [0, 1]
        return self.linear(self.cnn(observations.float() / 255.0))


# ── Evaluation callback ───────────────────────────────────────────────────────

class WinRateCallback(BaseCallback):
    def __init__(self, eval_env, eval_episodes=10000, eval_freq=500_000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq < self.training_env.num_envs:
            wins, episodes = 0, 0
            obs = self.eval_env.reset()
            while episodes < self.eval_episodes:
                action, _ = self.model.predict(obs, deterministic=False)
                obs, _, dones, infos = self.eval_env.step(action)
                for done, info in zip(dones, infos):
                    if done:
                        episodes += 1
                        if info.get("episode", {}).get("r", 0) >= coinrun_win_reward:
                            wins += 1
            win_rate = wins / self.eval_episodes
            self.logger.record("eval/test_win_rate", win_rate)
            if self.verbose:
                print(f"\n[{self.num_timesteps:,} steps] Test Win Rate: {win_rate*100:.1f}%")
        return True


# ── Learning rate schedule ────────────────────────────────────────────────────

def linear_schedule(initial_value: float, final_value: float = 2e-4) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func


# ── Main ──────────────────────────────────────────────────────────────────────

def make_env(
    start_level: int,
    num_levels: int,
    num_envs: int = 256,
    shape_rewards: bool = True,
    time_penalty: float = 0.01,
) -> VecMonitor:
    """Build a stacked ProcgenEnv → RGB → [RewardShaping] → VecMonitor pipeline.

    shape_rewards should be True for training and False for evaluation so that
    VecMonitor records raw episode returns and the win-rate check (>= 10.0) works.
    """
    raw_env = ProcgenEnv(
        num_envs=num_envs,
        env_name="coinrun",
        start_level=start_level,
        num_levels=num_levels,
        distribution_mode="easy",
        rand_seed=global_seed,
    )
    env = ProcgenRGBWrapper(raw_env)
    if shape_rewards:
        env = RewardShapingWrapper(env, time_penalty=time_penalty)
    env = VecMonitor(env)
    return env


def main():
    num_envs = 256
    model_name = "impala_coinrun_customreward_25m_easy_customreward0.01_seed1"
    print(f"Initializing {num_envs} train + {num_envs} eval environments...")

    train_env = make_env(start_level=0,                num_levels=0,               num_envs=num_envs, shape_rewards=True)
    eval_env  = make_env(start_level=eval_start_level, num_levels=eval_num_levels, num_envs=num_envs, shape_rewards=False)  # raw rewards so episode.r >= 10.0 win detection works

    # we pass normalize_images=False so SB3 doesn't divide by 255 before our forward()
    policy_kwargs = dict(
        features_extractor_class=ImpalaCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,
    )

    model = PPO(
        "CnnPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=linear_schedule(5e-4),
        n_steps=256,
        batch_size=4096,
        n_epochs=6,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.001,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cuda",
        tensorboard_log="./final_coinrun_tensorboard",
    )

    callback = WinRateCallback(
        eval_env=eval_env,
        eval_episodes=1_000,
        eval_freq=500_000,
        verbose=1,
    )

    total_timesteps = 25_000_000
    print(f"Starting training for {total_timesteps:,} steps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name=model_name, callback=callback)

    model.save(model_name)
    print("Training complete!")


if __name__ == "__main__":
    main()
