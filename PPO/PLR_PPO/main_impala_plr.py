import gymnasium as gym
import torch as th
import torch.nn as nn
import numpy as np

import glob
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, VecEnvWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import spaces
from procgen import ProcgenEnv, ProcgenGym3Env
from typing import Callable

eval_start_level   = 100_000
eval_num_levels    = 1_000
coinrun_win_reward = 10.0
global_seed        = 1

# impala cnn architecture
# https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/master/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        # people call this a resnet style skip connector, input is added to the output of the conv layers
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

        # SB3 feeds images as (C, H, W) channels first
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            ImpalaBlock(n_input_channels, 16),
            ImpalaBlock(16, 32),
            ImpalaBlock(32, 32),
            nn.ReLU(),
            nn.Flatten(),
        )

        # infer the flattened size with a dummy forward pass
        # this is done so that we don't have to manually calculate the size of the flattened output
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

"""
    PLR BUFFER
    It's a scoreboard that tracks three arrays over 500 levels. These three are:
    
    scores: Represents how much the agent has still to learn from that specific level
    timestamps: When was the level last visited
    seen flag: Whether the level has been previously visited or not
    
    There's two methods in this class: 
    
    update_score(): stores the new scores after an episode
    sample_level(): implements the PLR sampling logic. In the beginning it tries to visit unseen levels
    with a probability that decreases the more levels are visited. Later it samples using the ranked and staleness based 
    score distribution. 
"""
class PLRBuffer:
    def __init__(self, num_levels: int = 500, beta: float = 0.1, rho: float = 0.1):
        self.num_levels    = num_levels
        self.beta          = beta
        self.rho           = rho
        self.scores        = np.zeros(num_levels, dtype=np.float32)
        self.timestamps    = np.zeros(num_levels, dtype=np.float32)
        self.seen          = np.zeros(num_levels, dtype=bool)
        self.episode_count = 0

    def update_score(self, level_idx: int, l1_value_loss: float):
        self.scores[level_idx]     = l1_value_loss
        self.timestamps[level_idx] = self.episode_count
        self.seen[level_idx]       = True

    def sample_level(self) -> int:
        self.episode_count += 1
        unseen = np.where(~self.seen)[0]

        explore_prob = len(unseen) / self.num_levels
        if len(unseen) > 0 and np.random.random() < explore_prob:
            return int(np.random.choice(unseen))

        seen_idx = np.where(self.seen)[0]
        if len(seen_idx) == 0:
            return int(np.random.randint(self.num_levels))

        seen_scores     = self.scores[seen_idx]
        ranks           = np.argsort(np.argsort(-seen_scores)) + 1
        score_w         = (1.0 / ranks) ** (1.0 / self.beta)
        score_probs     = score_w / score_w.sum()

        staleness       = self.episode_count - self.timestamps[seen_idx]
        staleness_probs = staleness / staleness.sum() if staleness.sum() > 0 \
                          else np.ones(len(seen_idx)) / len(seen_idx)

        probs = (1 - self.rho) * score_probs + self.rho * staleness_probs
        return int(np.random.choice(seen_idx, p=probs / probs.sum()))

# https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
class ProcgenRGBWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(15)

    def reset(self):
        obs = self.venv.reset()
        return obs["rgb"]

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs["rgb"], rewards, dones, infos

""" 
    PLR VecEnv
    It wraps ProcgenGym3Env instead of ProcgenEnv, which gives access to get_state and set_state.
    Two methods which are necessary for the PLR logic because we need to be able to tell a specific environment to
    replay a certain level.
    At startup, it pre fills a cache with the initial state of all 500 levels to be able to select a specific level to 
    replay. The cache is filled using the get_state method previously mentioned.
    Then we have 2 more methods:
    assign_levels() used when an episode ends. It only loops through the environment that finished 
    assign_all_levels() used once at the very start. It loops through all the 256 environments.
    Using both is more efficient because you don't want to interrupt every env that is mid-episode whenever one of 
    them finishes.
    
"""
class PLRVecEnv(VecEnv):

    def __init__(self, num_envs: int, env_name: str, num_levels: int,
                 plr_buffer: PLRBuffer, distribution_mode: str = "hard",
                 rand_seed: int = 1):
        super().__init__(
            num_envs,
            gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            gym.spaces.Discrete(15),
        )
        self.plr      = plr_buffer
        self._actions = None

        # rand_seed passed through so levels are identical to your baseline
        self._g3 = ProcgenGym3Env(
            num=num_envs, env_name=env_name,
            start_level=0, num_levels=num_levels,
            distribution_mode=distribution_mode,
            rand_seed=rand_seed,
        )

        print(f"Caching {num_levels} level states (startup only)...")
        self._level_cache = {}
        for lvl in range(num_levels):
            tmp = ProcgenGym3Env(
                num=1, env_name=env_name,
                start_level=lvl, num_levels=1,
                distribution_mode=distribution_mode,
                rand_seed=rand_seed,
            )
            self._level_cache[lvl] = tmp.callmethod("get_state")[0]
            tmp.close()
        print("Done caching.")

        self._env_levels = np.zeros(num_envs, dtype=int)
        self._assign_all_levels()

    def _assign_all_levels(self):
        all_states = self._g3.callmethod("get_state")
        for i in range(self.num_envs):
            lvl                 = self.plr.sample_level()
            all_states[i]       = self._level_cache[lvl]
            self._env_levels[i] = lvl
        self._g3.callmethod("set_state", all_states)

    def _assign_levels(self, env_indices):
        all_states = self._g3.callmethod("get_state")
        for i in env_indices:
            lvl                 = self.plr.sample_level()
            all_states[i]       = self._level_cache[lvl]
            self._env_levels[i] = lvl
        self._g3.callmethod("set_state", all_states)

    def reset(self):
        self._assign_all_levels()
        self._g3.act(np.zeros(self.num_envs, dtype=int))
        _, obs, _ = self._g3.observe()
        return obs["rgb"]

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        self._g3.act(self._actions)
        rew, obs, first = self._g3.observe()

        finished = np.where(first)[0]
        if len(finished) > 0:
            self._assign_levels(finished)
            _, obs, _ = self._g3.observe()

        dones = first.astype(bool)
        infos = [{"level": int(self._env_levels[i])} for i in range(self.num_envs)]
        return obs["rgb"], rew, dones, infos

    def close(self):
        self._g3.close()

    def get_attr(self, attr_name, indices=None):               return [None] * self.num_envs
    def set_attr(self, attr_name, value, indices=None):        pass
    def env_method(self, method_name, *a, indices=None, **kw): return [None] * self.num_envs
    def env_is_wrapped(self, wrapper_class, indices=None):     return [False] * self.num_envs
    def seed(self, seed=None):                                 return [None] * self.num_envs

""" 
    PLR CALLBACK
    This class is used to connect the SB3 rollout buffer with the PLR scoring system.
    It has three methods: 
    
    _on_rollout_start(): takes a snapshot of which levels each environments are currently on. This is necessary because 
    environments get assigned new levels mid-rollout when episodes end so without the snapshot, the scores would be
    assigned to the wrong levels.
    _on_rollout_end(): Reads values (Predictions), returns (Targets), and episode_starts (Boolean used to know if we're
    at the beginning of the episode or not and we need this to find the boundaries between episodes) from SB3's rollout 
    buffer. In addition to reading these, it computes L1 values loss per episode and updates the PLR scores.
    _on_step(): Returns True as required by SB3 to keep training
"""
class PLRCallback(BaseCallback):
    def __init__(self, plr_env: PLRVecEnv, verbose: int = 0):
        super().__init__(verbose)
        self.plr_env = plr_env
        self._rollout_start_levels = None

    def _on_rollout_start(self):
        self._rollout_start_levels = self.plr_env._env_levels.copy()

    def _on_rollout_end(self):
        buf = self.model.rollout_buffer
        n_steps, n_envs = self.model.n_steps, self.training_env.num_envs

        values  = buf.values.reshape(n_steps, n_envs)
        returns = buf.returns.reshape(n_steps, n_envs)
        starts  = buf.episode_starts.reshape(n_steps, n_envs)

        for env_i in range(n_envs):
            current_level = self._rollout_start_levels[env_i]
            ep_start = 0
            for t in range(n_steps):
                if t > 0 and starts[t, env_i]:
                    ep_loss = np.abs(
                        values[ep_start:t, env_i] - returns[ep_start:t, env_i]
                    ).mean()
                    self.plr_env.plr.update_score(int(current_level), float(ep_loss))
                    current_level = self.plr_env._env_levels[env_i]
                    ep_start = t

    def _on_step(self) -> bool:
        return True

class WinRateCallback(BaseCallback):
    def __init__(self, eval_env, eval_episodes=1000, eval_freq=500_000, verbose=1):
        super().__init__(verbose)
        self.eval_env      = eval_env
        self.eval_episodes = eval_episodes
        self.eval_freq     = eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq < self.training_env.num_envs:
            wins, episodes = 0, 0
            obs = self.eval_env.reset()
            while episodes < self.eval_episodes:
                action, _ = self.model.predict(obs, deterministic=True)
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

def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func

def get_latest_checkpoint(model_name):
    checkpoints = glob.glob(f"checkpoints/{model_name}_*_steps.zip")
    if not checkpoints:
        # also check current directory for backwards compatibility
        checkpoints = glob.glob(f"{model_name}_*_steps.zip")
    if not checkpoints:
        return None
    # sort numerically by step count
    checkpoints.sort(key=lambda x: int(x.split("_steps")[0].split("_")[-1]))
    return checkpoints[-1]

def main():
    NUM_ENVS    = 256
    NUM_LEVELS  = 500
    MODEL_NAME  = "impala_plr_coinrun"
    TOTAL_STEPS = 200_000_000

    print(f"Initializing {NUM_ENVS} train + {NUM_ENVS} eval environments...")

    plr_buf   = PLRBuffer(num_levels=NUM_LEVELS, beta=0.1, rho=0.1)
    plr_env   = PLRVecEnv(
        num_envs=NUM_ENVS, env_name="coinrun",
        num_levels=NUM_LEVELS, plr_buffer=plr_buf,
        distribution_mode="hard", rand_seed=global_seed,
    )
    train_env = VecMonitor(plr_env)

    eval_env = VecMonitor(ProcgenRGBWrapper(
        ProcgenEnv(num_envs=NUM_ENVS, env_name="coinrun",
                   start_level=eval_start_level,
                   num_levels=eval_num_levels,
                   distribution_mode="hard",
                   rand_seed=global_seed)
    ))

    callbacks = [
        PLRCallback(plr_env, verbose=0),
        WinRateCallback(eval_env, eval_episodes=1_000,
                        eval_freq=500_000, verbose=1),
	    # Here the save_freq can't simply be a simple number because it counts in step() calls not env calls and 1
        # step() call is equal to 256 env steps.
	    CheckpointCallback(save_freq = 5_000_000 // 256,
                           save_path = "./", name_prefix = MODEL_NAME, verbose = 1),
    ]

    latest = get_latest_checkpoint(MODEL_NAME)
    if latest:
        print(f"Checkpoint found: {latest} — resuming training...")
        model = PPO.load(latest, env=train_env, device="cuda")
        remaining = TOTAL_STEPS - model.num_timesteps
        print(f"Resuming from {model.num_timesteps:,} steps, {remaining:,} remaining...")
    else:
        print("No checkpoint found — starting fresh...")
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
            learning_rate=5e-4,
            n_steps=256,
            batch_size=8192,
            n_epochs=3,
            gamma=0.999,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=1.0,
            device="cuda",
            tensorboard_log="./final_coinrun_tensorboard",
        )
        remaining = TOTAL_STEPS

    if remaining <= 0:
        print("Already reached 200M steps, nothing to do.")
        return

    print(f"Training for {remaining:,} remaining steps...")
    model.learn(
        total_timesteps=remaining,
        progress_bar=True,
        tb_log_name=MODEL_NAME,
        callback=callbacks,
        reset_num_timesteps=False,
    )
    model.save(MODEL_NAME)
    print("Training complete!")


if __name__ == "__main__":
    main()
