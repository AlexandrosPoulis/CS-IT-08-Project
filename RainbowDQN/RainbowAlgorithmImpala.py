# .\rl-project-gymnasium\Scripts\activate

# The python version needed is 3.13 for the procgen_gym

import gymnasium as gym
#from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv
import procgen_gym  # gymnasium procgen version(procgen is in gym causing problems with AgileRL)
#from typing import Callable
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from agilerl.algorithms.dqn_rainbow import RainbowDQN
from agilerl.training.train_off_policy import train_off_policy
from agilerl.components.replay_buffer import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
)
from agilerl.wrappers.make_evolvable import MakeEvolvable
# This would be used if the infinite error problem was not encountered.
from impala_cnn_for_rainbow import ImpalaCNN
from agilerl.networks.q_networks import RainbowQNetwork
from gymnasium.vector import AsyncVectorEnv
from agilerl.networks.q_networks import RainbowQNetwork

# this is to suppress some warnings.
import warnings
os.environ['WANDB_SILENT'] = 'true'
warnings.filterwarnings('ignore', category=RuntimeWarning)



class ProcgenPreprocess(gym.ObservationWrapper):
    # A wrapper needed to adjust the procgen observation space to be compatible 
    # with the ImpalaCNN architecture and AgileRL's expected input format.

    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(c, h, w), dtype=np.float32
        )

    def observation(self, obs):
        obs = np.transpose(obs, (2, 0, 1))
        return obs.astype(np.float32)

class SingleEnvTensorboardLogger(gym.Wrapper):
    def __init__(self, env, env_id, log_dir="./tensorboard_logs"):
        super().__init__(env)
        self.writer = SummaryWriter(log_dir)
        self.env_id = env_id
        self.episode_reward = 0.0
        self.episode_length = 0
        self.global_step = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_length += 1
        self.global_step += 1
        
        if terminated or truncated:
            self.writer.add_scalar(f"Rollout/Episode_Length/Env_{self.env_id}", self.episode_length, self.global_step)
            # You can also keep tracking reward here if you want to see per-env variance!
            self.writer.add_scalar(f"Rollout/Episode_Reward/Env_{self.env_id}", self.episode_reward, self.global_step)
            self.episode_reward = 0.0
            self.episode_length = 0
            
        return obs, reward, terminated, truncated, info

def find_latest_checkpoint(folder_path, extension=".pt"):
    # Function to find the latest checpoint and continue training from that
    if not os.path.exists(folder_path):
        return None
    
    max_size = 0
    max_file = None
    
    for folder, _, files in os.walk(folder_path):
        for file in files:
            if not file.endswith(extension):
                continue
                
            full_path = os.path.join(folder, file)
            try:
                size = os.stat(full_path).st_size
                if size > max_size:
                    max_size = size
                    max_file = full_path
            except OSError:
                continue
    
    return max_file

def make_env(env_id=0):
    env_start_id = 10
    env = gym.make("procgen_gym/procgen-coinrun-v0", 
                    num_levels=50,
                    start_level=0,
                    distribution_mode="easy",
                    #render_mode="rgb_array",#
                    rand_seed = env_start_id)
    env = ProcgenPreprocess(env)
    env = SingleEnvTensorboardLogger(env, env_id)

    return env


class ImpalaRainbowQNetwork(RainbowQNetwork):

    #Custom Q-network to connect Imapla to AgileRL
    def __init__(
        self,
        observation_space,
        action_space,
        support,
        num_atoms=51,
        noise_std=0.5,
        encoder_config=None,
        head_config=None,
        min_latent_dim=8,
        max_latent_dim=256,
        latent_dim=256,
        device="cpu",
        random_seed=None,
    ):
        # Initialize the RainbowQNetwork class.
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            support=support,
            num_atoms=num_atoms,
            noise_std=noise_std,
            # this is not used its to initialize with a simple encoder_cofig
            encoder_config={"channel_size": [32], "kernel_size": [8], "stride_size": [4]},
            head_config=head_config,
            min_latent_dim=min_latent_dim,
            max_latent_dim=max_latent_dim,
            latent_dim=latent_dim,
            device=device,
            random_seed=random_seed,
        )
        
        # This part is needed because of a infinite error problem, in newer versions might not be nessesary
        if hasattr(self, "_evolvable_modules") and "encoder" in self._evolvable_modules:
            del self._evolvable_modules["encoder"]
        self.filter_mutation_methods("encoder.")
        
        # Use the ImpalaCNN file we have made
        self.encoder_cls = ImpalaCNN
        self.encoder = ImpalaCNN(observation_space, features_dim=latent_dim, device=device).to(device)

# Global variable for seting up the folder where the models will be sent to/model will be 
# loaded from
EXPERIMENT_NAME = "Rainbow_HardMode_Test4__test"

def train_agent(resume_from_checkpoint=None):
    base_dir = f"./experiments/{EXPERIMENT_NAME}"
    os.makedirs(f"{base_dir}/checkpoints", exist_ok=True)
    
    # wandb sync --local ./experiments/Rainbow_Batch128_EasyMode_Test1/wandb/latest-run
    wb_settings = {
        "project": "Procgen_CoinRun",
        "addl_args": {
            "name": EXPERIMENT_NAME,
            "dir": base_dir,
            "mode": "offline", # set to online to automatically sent everything to the cloud.
            "sync_tensorboard": True,
            "resume": "allow" 
        }
    }

    # Check if cuda is available(GPU) if not use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device used = {device}")
    # This can be set higher if using a GPU
    nums_envs = 8

    env = AsyncVectorEnv([lambda i=i: make_env(i) for i in range(nums_envs)])

    # The hyperparameters for the Rainbow DQN agent.
    INIT_HP = {
        "BATCH_SIZE": 128,  # Batch size
        "LR": 0.0005,  # Learning rate
        "GAMMA": 0.999,  # Discount factor
        "MEMORY_SIZE": 50_000,  # Max memory buffer size
        "LEARN_STEP": 8 * nums_envs,  # Learning frequency
        "N_STEP": 7,  # Step number to calculate td error
        "PER": True,  # Use prioritized experience replay buffer
        "ALPHA": 0.7,  # Prioritized replay buffer parameter 
        "BETA": 0.4,  # Importance sampling coefficient 
        "TAU": 0.01,  # For soft update of target parameters 
        "PRIOR_EPS": 0.000001,  # Minimum priority for sampling
        "NUM_ATOMS": 51,  # Unit number of support
        "V_MIN": 0.0,  # Minimum value of support
        "V_MAX": 10.0,  # Maximum value of support
        "NOISY": True,  # Add noise directly to the weights of the network
        "LEARNING_DELAY": 20_000,  # Steps before starting learning
        "TARGET_SCORE": 10.0,  # Target score that will beat the environment
        "MAX_STEPS": 50_000_000,  # Maximum number of steps an agent takes in an environment
        "EVO_STEPS": 20_000,  # Evolution frequency
        "EVAL_STEPS": None,  # Number of evaluation steps per episode
        "EVAL_LOOP": 1,  # Number of evaluation episodes
        # The comments on these are left the same as the ones used by AgileRL's documentation
        # to be more accurate
    }

    # support tensor required by Rainbow (distributional RL)
    support = torch.linspace(
        INIT_HP["V_MIN"], INIT_HP["V_MAX"], INIT_HP["NUM_ATOMS"]
    ).to(device)

    actor_network = ImpalaRainbowQNetwork(
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        support=support,
        num_atoms=INIT_HP["NUM_ATOMS"],
        latent_dim=256,
        device=device,
    )
    rainbow_dqn = RainbowDQN(
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        actor_network=actor_network,
        batch_size=INIT_HP["BATCH_SIZE"],
        lr=INIT_HP["LR"],
        learn_step=INIT_HP["LEARN_STEP"],
        gamma=INIT_HP["GAMMA"],
        tau=INIT_HP["TAU"],
        beta=INIT_HP["BETA"],
        n_step=INIT_HP["N_STEP"],
        device=device, 
    )

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

    # Used to check if training will be resumed or a new model will be trained
    if resume_from_checkpoint:
        rainbow_dqn = RainbowDQN.load(f"{base_dir}/checkpoints/{resume_from_checkpoint}", device=device)
        
        # Move the full agent to device
        rainbow_dqn.actor = rainbow_dqn.actor.to(device)
        rainbow_dqn.actor_target = rainbow_dqn.actor_target.to(device)
        
        # Explicitly move the ImpalaCNN encoder inside both networks
        rainbow_dqn.actor.encoder = rainbow_dqn.actor.encoder.to(device)
        rainbow_dqn.actor_target.encoder = rainbow_dqn.actor_target.encoder.to(device)
        
        # Move support tensor
        if hasattr(rainbow_dqn, 'support'):
            rainbow_dqn.support = rainbow_dqn.support.to(device)
        
        # Move optimizer state
        for state in rainbow_dqn.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        print(f"Resumed from checkpoint. Actor encoder device: {next(rainbow_dqn.actor.encoder.parameters()).device}")
        # Loaded Rainbow DQN agent's hyperparameters
        # mostly  here to remind the user in case notes about 
        # a test are  lost
        print(f"lr: {rainbow_dqn.lr}")
        print(f"gamma: {rainbow_dqn.gamma}")
        print(f"batch_size: {rainbow_dqn.batch_size}")
        print(f"learn_step: {rainbow_dqn.learn_step}")
        print(f"n_step: {rainbow_dqn.n_step}")
        print(f"tau: {rainbow_dqn.tau}")
        print(f"beta: {rainbow_dqn.beta}")
        print(f"prior_eps: {rainbow_dqn.prior_eps}")
        print(f"noise_std: {rainbow_dqn.noise_std}")


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
        wb=True, # Boolean flag to record run with Weights & Biases
        wandb_kwargs=wb_settings,
        checkpoint=INIT_HP["MAX_STEPS"]//50,
        checkpoint_path=f"{base_dir}/checkpoints/checkpoint.pt",
        save_elite=True#,
        #elite_path=f"{base_dir}/best_agent.pt"
    )
    


if __name__ == '__main__':
    checkpoint_folder = f"./experiments/{EXPERIMENT_NAME}/checkpoints"
    
    # Auto-detect latest checkpoint
    latest_checkpoint = find_latest_checkpoint(checkpoint_folder)
    
    if latest_checkpoint:
        checkpoint_name = os.path.basename(latest_checkpoint)
        print(f"Found latest checkpoint: {checkpoint_name}")
        train_agent(checkpoint_name)
    else:
        print("No checkpoint found. Training from scratch...")
        train_agent(None)

