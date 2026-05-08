# .\rl-project-gymnasium\Scripts\activate

# The python version needed is 3.13 for the procgen_gym

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import procgen_gym  # gymnasium procgen version(procgen is in gym causing problems with AgileRL)
from typing import Callable
import torch
import numpy as np
import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from agilerl.algorithms.dqn_rainbow import RainbowDQN
from agilerl.training.train_off_policy import train_off_policy
from agilerl.components.replay_buffer import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
)
from gymnasium.vector import AsyncVectorEnv

class ProcgenPreprocess(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(c, h, w), dtype=np.float32#0.0, 1.0
        )

    def observation(self, obs):
        obs = np.transpose(obs, (2, 0, 1))
        return obs.astype(np.float32)
    

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def print_env_structure(env):
    # This function returns some values that could be helpfull 
    # for one to understand the structure of things such as the action space,
    # observation space and a few others. 

    obs, info = env.reset()
    print(env.action_space)
    print(obs.shape)
    print(env.single_observation_space)
    print(type(env.single_observation_space))
    print(env.single_observation_space.shape)

def make_env():
    env = gym.make("procgen_gym/procgen-coinrun-v0", 
                    num_levels=500,
                    start_level=0,
                    distribution_mode="hard",
                    #render_mode="rgb_array",#
                    rand_seed = 1)
    #env = FrameStackObservation(env, 4)
    return env
    #env = gym.make(..., frame_skip=2)

def train_agent():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device used = {device}")
    nums_envs = 16

    env = SyncVectorEnv([make_env for _ in range(nums_envs)])
    env = ProcgenPreprocess(env)
    #env = AsyncVectorEnv([lambda: make_env() for _ in range(nums_envs)])
    #env = SyncVectorEnv([make_env for _ in range(nums_envs)])


    print_env_structure(env)

    # The CNN used. Taken from the AgileRl Rainbow & cartpole tutorial
    net_config = {
        "encoder_config": {
            "channel_size": [32, 32], # [32,64] might be better for feature extraction
            "kernel_size": [8, 4],
            "stride_size": [4, 2],
        },
        "head_config": {
            "hidden_size": [32], #128 might be better   
        }
    }
    # The hyperparameters used, taken from the AgileRl Rainbow &  cartpole tutorial found in the documentation
    # and adjusted with some of the values sudgested by the original Rainbow DQN paper 
    INIT_HP = {
        "BATCH_SIZE": 32,  # Batch size
        "LR": 0.0001,  # Learning rate
        "GAMMA": 0.99,  # Discount factor
        "MEMORY_SIZE": 200_000,  # Max memory buffer size
        "LEARN_STEP": 5,  # Learning frequency
        "N_STEP": 3,  # Step number to calculate td error
        "PER": True,  # Use prioritized experience replay buffer
        "ALPHA": 0.5,  # Prioritized replay buffer parameter #0.6
        "BETA": 0.5,  # Importance sampling coefficient #0.4
        "TAU": 0.001,  # For soft update of target parameters
        "PRIOR_EPS": 0.000001,  # Minimum priority for sampling
        "NUM_ATOMS": 51,  # Unit number of support
        "V_MIN": -10.0,  # Minimum value of support
        "V_MAX": 10.0,  # Maximum value of support
        "NOISY": True,  # Add noise directly to the weights of the network
        "LEARNING_DELAY": 80000,  # Steps before starting learning
        "TARGET_SCORE": 10.0,  # Target score that will beat the environment
        "MAX_STEPS": 10_000_000,  # Maximum number of steps an agent takes in an environment
        "EVO_STEPS": 10000,  # Evolution frequency
        "EVAL_STEPS": None,  # Number of evaluation steps per episode
        "EVAL_LOOP": 1,  # Number of evaluation episodes
    }

    rainbow_dqn = RainbowDQN(
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        net_config=net_config,
        batch_size=INIT_HP["BATCH_SIZE"],
        lr=INIT_HP["LR"],
        learn_step=INIT_HP["LEARN_STEP"],
        gamma=INIT_HP["GAMMA"],
        tau=INIT_HP["TAU"],
        beta=INIT_HP["BETA"],
        n_step=INIT_HP["N_STEP"],
        device=device, #memory_device, #device,
    )

    rainbow_dqn.set_training_mode(True)
    memory = PrioritizedReplayBuffer(
        max_size=INIT_HP["MEMORY_SIZE"],
        alpha=INIT_HP["ALPHA"],
        device=device,
    )
    n_step_memory = MultiStepReplayBuffer(
        max_size=INIT_HP["MEMORY_SIZE"],
        n_step=INIT_HP["N_STEP"],
        gamma=INIT_HP["GAMMA"],
        device=device,
    )
    trained_pop, pop_fitnesses = train_off_policy(
        env=env,
        env_name="coinrun",
        algo="RainbowDQN",
        pop=[rainbow_dqn],
        memory=memory,
        n_step_memory=n_step_memory,
        max_steps=INIT_HP["MAX_STEPS"],
        evo_steps=INIT_HP["EVO_STEPS"],
        learning_delay=INIT_HP["LEARNING_DELAY"],
        n_step=True,
        per=True,
        wb=False, # Boolean flag to record run with Weights & Biases
        checkpoint=500_000,
        checkpoint_path="RainbowDQN.pt"
    )
    


if __name__ == '__main__':
    train_agent()
