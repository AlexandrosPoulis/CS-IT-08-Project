import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecMonitor, VecEnvWrapper
from procgen import ProcgenEnv
import numpy as np
from typing import Callable

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

def linear_schedule(initial_value: float, final_value: float = 2e-4) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func
model_name = "lstm"
def main():
    print("Initializing 256 parallel environments...")
    env = ProcgenEnv(
        num_envs=256,
        env_name="coinrun",
        start_level=0,
        num_levels=200000,
        distribution_mode="hard"
    )
    env = ProcgenRGBWrapper(env)
    env = VecMonitor(env)

    model = RecurrentPPO(
    "CnnLstmPolicy",
    env,
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
        max_grad_norm=0.5,
        device="cuda",
        tensorboard_log="./comp_coinrun_tensorboard"
    )

    total_timesteps = 50_000_000
    print(f"Starting training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(model_name)
    print("Training complete!")

if __name__ == '__main__':
    main()