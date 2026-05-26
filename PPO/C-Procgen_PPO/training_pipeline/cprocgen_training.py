import gymnasium as gym
import numpy as np
import os
import json
import imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecEnvWrapper, VecNormalize
from stable_baselines3.common.logger import configure
from typing import Callable
from cprocgen import CProcgenEnv


# Contexts (NEW)
context_options = [
    {"visibility": 9,
    "allow_monsters": True}
]

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

def main():
    hyperparams = {
        "env_name": "coinrun",
        "num_envs": 64,
        "policy_type": "CnnPolicy",
        "total_timesteps": 256_000_000,   
        "learning_rate": 5e-4,
        "n_steps": 256,
        "batch_size": 2048,
        "n_epochs": 3,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "device": "cuda",
    }

    print("Initializing C-Procgen environments...")

    env = CProcgenEnv(
        num_envs=hyperparams["num_envs"],
        env_name=hyperparams["env_name"],
        start_level=0,    
        num_levels=500,
        context_options=context_options,
    )

    env = CProcgenRGBWrapper(env)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True)


    # Logging setup
    timesteps_m = hyperparams["total_timesteps"] // 1_000_000
    base_model_name = f"{hyperparams['policy_type']}_{timesteps_m}M_CProcgen"

    model_root = os.path.join("tensorboard_logs", "cprocgen", "models")
    run_name, run_dir, model_path = get_run_paths(base_model_name, model_root)

    print("Run folder:", run_dir)
    print("Model will be saved as:", model_path)
    
    tensorboard_dir = run_dir
    log_data = {
        "hyperparameters": hyperparams,
        "context_options": context_options,  
        "run_name": run_name,
        "model_path": model_path,
        "tensorboard_dir": tensorboard_dir,
    }

    with open(os.path.join(tensorboard_dir, "training_config.json"), "w") as f:
        json.dump(log_data, f, indent=4)
    
    # Model
    model = PPO(
        hyperparams["policy_type"],
        env,
        verbose=1,
        learning_rate=linear_schedule(hyperparams["learning_rate"]),
        n_steps=hyperparams["n_steps"],
        batch_size=hyperparams["batch_size"],
        n_epochs=hyperparams["n_epochs"],
        gamma=hyperparams["gamma"],
        gae_lambda=hyperparams["gae_lambda"],
        ent_coef=hyperparams["ent_coef"],
        clip_range=hyperparams["clip_range"],
        vf_coef=hyperparams["vf_coef"],
        device=hyperparams["device"],
    )

    model.set_logger(configure(run_dir, ["stdout", "tensorboard"]))

    # Train
    print(f"Starting training for {hyperparams['total_timesteps']} steps...")
    model.learn(total_timesteps=hyperparams["total_timesteps"], progress_bar=True)
    
    # Save
    model.save(model_path)
    env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    print("Training complete!")


if __name__ == "__main__":
    main()