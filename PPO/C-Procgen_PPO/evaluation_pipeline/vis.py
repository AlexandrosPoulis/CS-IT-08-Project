import json
import numpy as np
from collections import deque
from stable_baselines3 import PPO
from pathlib import Path
from procgen import CProcgenGym3Env
from gym3 import ViewerWrapper

# fix numpy compatibility
if "bool8" not in np.__dict__:
    np.bool8 = np.bool_

# Load model
project_root = Path(__file__).resolve().parents[1]
model_path = project_root / "CProcgen-cprocgen-release" / "tensorboard_logs" / "cprocgen" / "models" / "CnnPolicy_50M_coinrun_seed42_v5" / "CnnPolicy_50M_coinrun_seed42_v5"
model = PPO.load(str(model_path))

config_path = model_path.parent / "training_config.json"
frame_stack = 1
if config_path.exists():
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    frame_stack = int(config.get("hyperparameters", {}).get("frame_stack", 1))

# Contexts
context_options = [
    {"visibility": 13,
     "allow_monsters": True},
]

if config_path.exists():
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if isinstance(config.get("context_options"), list):
        context_options = config["context_options"]

class FrameStacker:
    def __init__(self, n_stack: int):
        self.n_stack = max(1, n_stack)
        self.frames = deque(maxlen=self.n_stack)

    def reset(self, obs: np.ndarray) -> np.ndarray:
        self.frames.clear()
        frame = self._to_chw(obs)
        for _ in range(self.n_stack):
            self.frames.append(frame)
        return self.get()

    def update(self, obs: np.ndarray, is_first: bool) -> np.ndarray:
        if is_first or len(self.frames) == 0:
            return self.reset(obs)
        self.frames.append(self._to_chw(obs))
        return self.get()

    def get(self) -> np.ndarray:
        stacked = np.concatenate(list(self.frames), axis=0)
        return stacked[None, ...]

    @staticmethod
    def _to_chw(obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 4:
            obs = obs[0]
        return np.transpose(obs, (2, 0, 1))


def decode_action(action_id: int) -> str:
    vx = action_id // 3 - 1
    vy = action_id % 3 - 1

    vx_label = {
        -1: "left",
        0: "idle",
        1: "right",
        2: "right_fast",
        3: "right_faster",
    }.get(vx, f"vx{vx}")

    vy_label = {
        -1: "down",
        0: "",
        1: "jump",
    }.get(vy, f"vy{vy}")

    if vy_label:
        if vx_label == "idle":
            return vy_label
        return f"{vx_label}+{vy_label}"
    return vx_label


# Create CProcgen env with contexts
env = CProcgenGym3Env(
    num=1,
    env_name="coinrun",
    start_level=1000000,
    num_levels=0,
    distribution_mode="easy",
    rand_seed=42,
    context_options=context_options,
)

# Viewer wrapper for human-like rendering
env = ViewerWrapper(env, ob_key="rgb")

# Evaluation
episodes_to_test = 100
episodes_completed = 0
wins = 0

print(f"Starting evaluation over {episodes_to_test} levels with contexts...")
print(f"Using frame_stack={frame_stack}")

stacker = FrameStacker(frame_stack)

try:
    rew, obs_dict, first = env.observe()
    obs = stacker.update(obs_dict["rgb"], bool(first[0]))

    while episodes_completed < episodes_to_test:
        # Predict action
        action, _ = model.predict(obs)
        action_id = int(action[0])
        print(f"Action: {action_id} ({decode_action(action_id)})")

        # Step environment
        env.act(action)

        # gym3 returns (reward, obs_dict, first)
        rew, obs_dict, first = env.observe()

        if first[0]:
            episodes_completed += 1
            if rew[0] > 0:
                wins += 1

            if episodes_completed % 10 == 0:
                print(f"Completed {episodes_completed}/{episodes_to_test} episodes...")

        obs = stacker.update(obs_dict["rgb"], bool(first[0]))

    win_probability = (wins / episodes_to_test) * 100

    print("\n--- Evaluation Results ---")
    print(f"Total Attempts: {episodes_to_test}")
    print(f"Total Wins: {wins}")
    print(f"Win Probability: {win_probability:.2f}%")

finally:
    if hasattr(env, "close"):
        env.close()
    elif hasattr(env, "env") and hasattr(env.env, "close"):
        env.env.close()