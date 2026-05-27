# The version of python needed to use is python 3.13+.
# The specific versions of procgen-gymnasium (procgen_gym) is  0.1.1 and for AgileRL is 2.6.1.

from argparse import Action

from argparse import Action

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import procgen_gym 


from typing import Callable
import torch
import numpy as np
from impala_cnn_for_rainbow import ImpalaCNN

from agilerl.algorithms.dqn_rainbow import RainbowDQN
from agilerl.training.train_off_policy import train_off_policy
from agilerl.components.replay_buffer import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
)
from gymnasium.vector import AsyncVectorEnv
from agilerl.utils.utils import make_vect_envs

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

   
def make_env(env_id=0):
    # Seed used for training was seed 1, seed for testing was decided to be 10. Both were used in the results to showcase, generalization.
    
    env_start_id = 10
    env = gym.make("procgen_gym/procgen-coinrun-v0", 
                    #num_levels=50, # 50 levels was used in the some training in others uncaped was used
                    start_level=0,
                    distribution_mode="hard",
                    #render_mode="human", # Uncomment if it is desired to see the evnironments and how
                    # the agent performs
                    rand_seed = env_start_id)
    env = ProcgenPreprocess(env)
    return env



def test():


    device = "cuda" if torch.cuda.is_available() else "cpu"
    rainbow_dqn = RainbowDQN.load('./experiments/Rainbow__HardMode_OgRainbowPapaerHyperparams/checkpoints/checkpoint_0_13019136.pt', device=device)
    # The model that had a non 0% winrate had this results
    # ./experiments/Rainbow__EasyMode_Test6/checkpoints/checkpoint_0_20027904.pt
    # easy uncaped
    #The terminated episodes were:10000                                                                                  
    #The truncated episodes were:0                                              
    #Totale is 10000
    #Win rate: 67.2% over 10000 episodes
    #Action distribution: {8: 638868}
    # RainbowDQN_0_3mill_50lvls.pt

    #easy 50 capped
    #The terminated episodes were:10000
    #The truncated episodes were:0
    #Totale is 10000
    #Win rate: 65.7% over 10000 episodes
    #Action distribution: {8: 624816}


    # hard 50 capped
    #The terminated episodes were:10001
    #The truncated episodes were:0
    #Totale is 10001
    #Win rate: 62.3% over 10001 episodes
    #Action distribution: {8: 631248}

    # hard uncapped
    #The terminated episodes were:10000                                                                 
    #The truncated episodes were:0                                         
    #Totale is 10000
    #Win rate: 60.8% over 10000 episodes
    #Action distribution: {8: 611880}


    nums_envs = 12
    test_env = AsyncVectorEnv([lambda i=i: make_env(i) for i in range(nums_envs)])

    episodes_to_test = 100
    episodes_completed = 0
    wins = 0
    tr_count = 0
    t_count = 0
    action_counts = {}
    obs, _ = test_env.reset()  

    # this is an evaluation loop.
    while episodes_completed < episodes_to_test:
        
        actions = rainbow_dqn.get_action(obs, training=False)
        
        for a in actions:
            action_counts[int(a)] = action_counts.get(int(a), 0) + 1

        obs, rewards, terminated, truncated, info = test_env.step(actions)

        for i, (r, t, tr) in enumerate(zip(rewards, terminated, truncated)):
            # this is to count how many episodes were
            # truncated and how many were ternimated in case 
            # it was needed.
            if tr:
                tr_count += 1
                break
            if t:
                t_count += 1
            
            if t or tr:
                episodes_completed += 1
                print(f"Num of episodes completed: {episodes_completed}")
                if r > 0:
                    wins += 1


    print(f"The terminated episodes were:{t_count}")
    print(f"The truncated episodes were:{tr_count}")
    print(f"Totale is {episodes_completed}")
    print(f"Win rate: {wins/max(episodes_completed,1)*100:.1f}% over {episodes_completed} episodes")
    print(f"Action distribution: {action_counts}")
    test_env.close()


if __name__ == '__main__':
    test()

# In AgileRL's documentation a way to make the testing into a gif was shown
# it used some of the bellow code left here in case such a showcase is needed in the future
# not in working order, mostly as a reminder
'''import os
import imageio
gif_path = "./videos/"
os.makedirs(gif_path, exist_ok=True)
imageio.mimwrite(
    os.path.join("./videos/", "rainbow_dqn_cartpole.gif"), frames, duration=10
)'''
