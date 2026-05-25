import torch
import torch.nn as nn
from gymnasium import spaces
#from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from agilerl.modules.base import EvolvableModule

# impala cnn architecture, followed our PPO implementation as a  baseline
# changed when needed to fit AgileRL and its reuirements.
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        
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


class ImpalaCNN(EvolvableModule): 
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, device="cpu"):
        super().__init__(device=device)
        self.observation_space = observation_space
        self.features_dim = features_dim
        self.name = "encoder"

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            ImpalaBlock(n_input_channels, 16),   
            ImpalaBlock(16, 32),                 
            ImpalaBlock(32, 32),                 
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor, hidden_state=None) -> torch.Tensor:
        x = observations.float()
        return self.linear(self.cnn(x))
        
    def get_init_dict(self):
        return {"observation_space": self.observation_space, "features_dim": self.features_dim, "device": self.device}

    def change_activation(self, activation: str, output: bool = False) -> None:
        pass

    @property
    def activation(self) -> str:
        return "ReLU"