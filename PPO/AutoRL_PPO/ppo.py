import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback
from procgen import ProcgenEnv
import numpy as np
from typing import Callable

eval_start_level = 100_000
eval_num_levels = 1_000
coinrun_win_reward = 10.0  


class ProcgenRGBWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(15)

    def reset(self):
        return self.venv.reset()["rgb"]

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs["rgb"], rewards, dones, infos


class WinRateCallback(BaseCallback):
    def __init__(self, eval_env, eval_episodes=1000, eval_freq=500_000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq

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

def main():
    nums_envs = 256
    model_name = "combinedv2"

    print(f"Initializing {nums_envs} train + {nums_envs} eval environments...")

    train_env = VecMonitor(ProcgenRGBWrapper(
        ProcgenEnv(num_envs=nums_envs, env_name="coinrun",start_level=0, num_levels=0, distribution_mode="hard")
    ))
    eval_env = VecMonitor(ProcgenRGBWrapper(
        ProcgenEnv(num_envs=nums_envs, env_name="coinrun", start_level=eval_start_level, num_levels=eval_num_levels, distribution_mode="hard")
    ))

    model = PPO(
        "CnnPolicy",
        train_env,
        verbose=1,
        learning_rate=5e-4,
        n_steps=256,
        batch_size=8192,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.001,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cuda",
        tensorboard_log="./final_coinrun_tensorboard"
    )

    callback = WinRateCallback(
        eval_env=eval_env,
        eval_episodes=1_000,
        eval_freq=500_000,
        verbose=1,
    )

    total_timesteps = 50_000_000
    print(f"Starting training for {total_timesteps:,} steps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name=model_name, callback=callback)
    model.save(model_name)
    print("Training complete!")

if __name__ == "__main__":
    main()