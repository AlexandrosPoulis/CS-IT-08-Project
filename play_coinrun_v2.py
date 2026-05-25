import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor
from procgen import ProcgenEnv
from typing import Callable

global_seed = 3
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

raw_env = ProcgenEnv(num_envs=512, env_name="coinrun", start_level=0, num_levels=0, distribution_mode="hard")
env = ProcgenRGBWrapper(raw_env)
env = VecMonitor(env)

model_path = "gamma_099_setseedv3"
model = PPO.load(model_path)

episodes_to_test = 50000
episodes_completed = 0
wins = 0

print(f"Starting evaluation of '{model_path}' over {episodes_to_test} unseen levels...")

obs = env.reset()
#obs, reward, done, info = env.step(np.array([env.action_space.sample()] * env.num_envs))
#print("Info dict contents:", info[0])
#import sys; sys.exit()

while episodes_completed < episodes_to_test:

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, info = env.step(action)

    for i, d in enumerate(done):
        if d:
            episodes_completed += 1

            # VecMonitor stores the full episode summary inside info under the "episode" key
            # We use an empty dict as fallback in case it is missing for any reason
            episode_info = info[i].get("episode", {})

            # "r" is the total reward accumulated over the episode
            # Any reward above 0 means the coin was collected, which counts as a win
            if episode_info.get("r", 0) > 0:
                wins += 1

        if episodes_completed % 10 == 0 and episodes_completed > 0:
            current_rate = (wins / episodes_completed) * 100
            print(f"Completed {episodes_completed}/{episodes_to_test} episodes... Win Rate: {current_rate:.1f}%")

win_probability = (wins / episodes_to_test) * 100

print("\n--- Final Evaluation Results ---")
print(f"Model Tested:                {model_path}")
print(f"Total Attempts:              {episodes_to_test}")
print(f"Total Wins (Coin Collected): {wins}")
print(f"Probability of Winning:      {win_probability:.2f}%")
env.close()