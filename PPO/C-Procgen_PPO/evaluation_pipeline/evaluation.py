import argparse
import csv
import json
from pathlib import Path
import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecFrameStack

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
	sys.path.insert(0, str(project_root))

from cprocgen import CProcgenEnv
eval_seeds = [10, 1234, 3333]

class ProcgenRGBWrapper(VecEnvWrapper):
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
		return obs


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Batch evaluate Procgen models")
	parser.add_argument(
		"--models_root",
		type=str,
		default="tensorboard_logs/ailab/gen_sweep/natureCNN",
		help="Root folder to search for training_config.json",
	)
	parser.add_argument("--episodes", type=int, default=10000)
	parser.add_argument("--eval_num_envs", type=int, default=512)
	parser.add_argument("--start_level", type=int, default=100000)
	parser.add_argument("--num_levels", type=int, default=0)
	parser.add_argument(
		"--output",
		type=str,
		default="results.csv",
	)
	parser.add_argument("--print_table", action="store_true")
	return parser.parse_args()


def resolve_model_path(project_root: Path, run_dir: Path, config: dict) -> Path | None:
	model_path_raw = config.get("model_path")
	candidate_paths = []

	if model_path_raw:
		model_path = Path(model_path_raw)
		if not model_path.is_absolute():
			model_path = project_root / model_path
		candidate_paths.append(model_path)

	# Prefer *_final.zip if it exists in the run directory
	final_candidates = sorted(run_dir.glob("*_final.zip"))
	candidate_paths = final_candidates + candidate_paths

	for path in candidate_paths:
		if path.exists():
			return path

	return None


def load_context_options(config: dict) -> list[dict]:
	if isinstance(config.get("context_options"), list):
		return config["context_options"]

	hyper = config.get("hyperparameters", {})
	raw = hyper.get("context_options")
	if isinstance(raw, str):
		try:
			return json.loads(raw)
		except json.JSONDecodeError:
			return []
	if isinstance(raw, list):
		return raw

	return []


def compute_training_win_rate_from_events(run_dir: Path) -> float | None:
	try:
		from tensorboard.backend.event_processing import event_accumulator
	except ImportError:
		print("TensorBoard is not installed; cannot read training win rate.")
		return None

	event_paths = sorted(run_dir.glob("events.out.tfevents.*"))
	if not event_paths:
		return None

	# Prefer latest event file for the most complete scalar history
	best_path = event_paths[-1]

	try:
		ea = event_accumulator.EventAccumulator(
			str(best_path),
			size_guidance={"scalars": 0},
		)
		ea.Reload()
		available_tags = ea.Tags().get("scalars", [])
		if "train/win_rate" in available_tags:
			scalars = ea.Scalars("train/win_rate")
			if scalars:
				return scalars[-1].value * 100.0
		if "train/win_rate_ema" in available_tags:
			scalars = ea.Scalars("train/win_rate_ema")
			if scalars:
				return scalars[-1].value * 100.0
		return None
	except Exception as exc:
		print(f"Failed to read event file {best_path}: {exc}")
		return None


def compute_training_ep_rew_mean_from_events(run_dir: Path) -> float | None:
	try:
		from tensorboard.backend.event_processing import event_accumulator
	except ImportError:
		print("TensorBoard is not installed; cannot read training ep_rew_mean.")
		return None

	event_paths = sorted(run_dir.glob("events.out.tfevents.*"))
	if not event_paths:
		return None

	best_path = event_paths[-1]

	try:
		ea = event_accumulator.EventAccumulator(
			str(best_path),
			size_guidance={"scalars": 0},
		)
		ea.Reload()
		available_tags = ea.Tags().get("scalars", [])
		if "rollout/ep_rew_mean" in available_tags:
			scalars = ea.Scalars("rollout/ep_rew_mean")
			if scalars:
				return scalars[-1].value
		return None
	except Exception as exc:
		print(f"Failed to read event file {best_path}: {exc}")
		return None


def decode_death_type(raw: object) -> str:
	death_type_map = {
		1: "saw",
		2: "enemy",
		3: "lava",
		4: "unknown",
		5: "timeout",
	}
	try:
		code = int(np.asarray(raw).item())
	except Exception:
		code = 4
	return death_type_map.get(code, "unknown")


def summarize_deaths(stats: dict) -> dict:
	total_deaths = stats.get("total_deaths", 0)
	if total_deaths <= 0:
		return {
			"saw": 0.0,
			"enemy": 0.0,
			"lava": 0.0,
			"timeout": 0.0,
			"unknown": 0.0,
		}
	return {
		"saw": (stats.get("saw", 0) / total_deaths) * 100.0,
		"enemy": (stats.get("enemy", 0) / total_deaths) * 100.0,
		"lava": (stats.get("lava", 0) / total_deaths) * 100.0,
		"timeout": (stats.get("timeout", 0) / total_deaths) * 100.0,
		"unknown": (stats.get("unknown", 0) / total_deaths) * 100.0,
	}


def load_training_failure_stats(run_dir: Path) -> dict:
	path = run_dir / "death_stats.json"
	if not path.exists():
		return {}
	try:
		with path.open("r", encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return {}


def evaluate_model(
	seed: int,
	model_path: Path,
	env_name: str,
	context_options: list[dict],
	frame_stack: int,
	episodes_to_test: int,
	eval_num_envs: int,
	start_level: int,
	num_levels: int,
) -> tuple[float, dict, float]:
	raw_env = CProcgenEnv(
		num_envs=eval_num_envs,
		env_name=env_name,
		context_options=context_options,
		start_level=start_level,
		num_levels=num_levels,
		rand_seed=seed,
	)

	env = ProcgenRGBWrapper(raw_env)
	env = VecMonitor(env)
	if frame_stack > 0:
		env = VecFrameStack(env, n_stack=frame_stack)

	model = PPO.load(model_path)

	episodes_completed = 0
	wins = 0
	reward_sum = 0.0
	death_stats = {
		"saw": 0,
		"enemy": 0,
		"lava": 0,
		"timeout": 0,
		"unknown": 0,
		"total_deaths": 0,
	}
	obs = env.reset()

	while episodes_completed < episodes_to_test:
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, done, info = env.step(action)
		done = np.asarray(done)
		for i, d in enumerate(done):
			if d:
				episodes_completed += 1

				episode_info = info[i].get("episode", {})
				ep_reward = episode_info.get("r", 0)
				reward_sum += float(ep_reward)

				prev_level_complete = info[i].get("prev_level_complete")
				if prev_level_complete is not None:
					try:
						is_win = bool(int(np.asarray(prev_level_complete).item()))
					except Exception:
						is_win = False
				else:
					is_win = ep_reward > 0

				if is_win:
					wins += 1
				else:
					death_stats["total_deaths"] += 1
					death_type = decode_death_type(info[i].get("death_type", 4))
					death_stats[death_type] += 1

			if episodes_completed >= episodes_to_test:
				break

	env.close()

	win_rate = (wins / episodes_to_test) * 100.0
	ep_rew_mean = reward_sum / max(1, episodes_to_test)
	return win_rate, death_stats, ep_rew_mean


def print_table(csv_path: Path) -> None:
	if not csv_path.exists():
		print(f"No evaluation file found at: {csv_path}")
		return

	with csv_path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		rows = list(reader)
		fieldnames = reader.fieldnames or []

	if not fieldnames:
		print(f"No columns found in: {csv_path}")
		return

	widths = {name: len(name) for name in fieldnames}
	for row in rows:
		for name in fieldnames:
			value = row.get(name, "")
			widths[name] = max(widths[name], len(str(value)))

	sep = "+" + "+".join("-" * (widths[name] + 2) for name in fieldnames) + "+"
	header = "|" + "|".join(f" {name.ljust(widths[name])} " for name in fieldnames) + "|"

	print(sep)
	print(header)
	print(sep)
	for row in rows:
		line = "|" + "|".join(
			f" {str(row.get(name, '')).ljust(widths[name])} " for name in fieldnames
		) + "|"
		print(line)
	print(sep)


def main() -> None:
	args = parse_args()
	project_root = Path(__file__).resolve().parents[1]
	models_root = project_root / args.models_root
	output_path = models_root / args.output

	config_paths = sorted(models_root.rglob("training_config.json"))
	if not config_paths:
		print(f"No training_config.json files found under: {models_root}")
		return

	output_path.parent.mkdir(parents=True, exist_ok=True)

	fieldnames = [
		"run_name",
		"env_name",
		"context",
		"train_win_rate",
		"train_ep_rew_mean",
		"train_death_saw_pct",
		"train_death_enemy_pct",
		"train_death_lava_pct",
		"train_death_timeout_pct",
		"train_death_unknown_pct",
		"test_win_rate_1",
		"test_win_rate_2",
		"test_win_rate_3",
		"test_win_rate_mean",
		"test_ep_rew_mean_1",
		"test_ep_rew_mean_2",
		"test_ep_rew_mean_3",
		"test_ep_rew_mean",
		"test_death_saw_pct",
		"test_death_enemy_pct",
		"test_death_lava_pct",
		"test_death_timeout_pct",
		"test_death_unknown_pct",
		"generalization_gap",
	]

	if output_path.exists():
		with output_path.open("r", encoding="utf-8") as f:
			first_line = f.readline().strip()
		if first_line and first_line.split(",")[0] != "run_name":
			existing_rows = output_path.read_text(encoding="utf-8")
			header_line = ",".join(fieldnames)
			output_path.write_text(f"{header_line}\n{existing_rows}", encoding="utf-8")

	completed_runs = set()
	if output_path.exists():
		with output_path.open("r", encoding="utf-8", newline="") as f:
			reader = csv.DictReader(f)
			for row in reader:
				run_name = row.get("run_name")
				if run_name:
					completed_runs.add(run_name)

	write_header = not output_path.exists()
	with output_path.open("a", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=fieldnames,
		)
		if write_header:
			writer.writeheader()

		for config_path in config_paths:
			run_dir = config_path.parent
			with config_path.open("r", encoding="utf-8") as cfg_file:
				config = json.load(cfg_file)

			run_name = config.get("run_name", run_dir.name)
			if run_name in completed_runs:
				print(f"Skipping (already evaluated): {run_name}")
				continue

			model_path = resolve_model_path(project_root, run_dir, config)
			if model_path is None:
				print(f"Skipping (model not found): {config_path}")
				continue

			hyper = config.get("hyperparameters", {})
			env_name = hyper.get("env_name", "coinrun")
			frame_stack = int(hyper.get("frame_stack", 0))
			context_options = load_context_options(config)
			context_label = json.dumps(context_options, ensure_ascii=True, separators=(",", ":"))

			print(f"Evaluating: {model_path}")

			train_failure_stats = load_training_failure_stats(run_dir)
			train_death_pct = summarize_deaths(train_failure_stats)

			test_win_rates = []
			test_ep_rew_means = []
			test_death_stats = {
				"saw": 0,
				"enemy": 0,
				"lava": 0,
				"timeout": 0,
				"unknown": 0,
				"total_deaths": 0,
			}
			for seed in eval_seeds:
				test_win_rate, seed_deaths, test_ep_rew_mean = evaluate_model(
					model_path=model_path,
					env_name=env_name,
					context_options=context_options,
					frame_stack=frame_stack,
					episodes_to_test=args.episodes,
					eval_num_envs=args.eval_num_envs,
					start_level=args.start_level,
					num_levels=args.num_levels,
					seed=seed,
				)
				test_win_rates.append(test_win_rate)
				test_ep_rew_means.append(test_ep_rew_mean)
				for key in ("saw", "enemy", "lava", "timeout", "unknown", "total_deaths"):
					test_death_stats[key] += seed_deaths.get(key, 0)

			test_win_rate_mean = sum(test_win_rates) / len(test_win_rates)
			test_ep_rew_mean = sum(test_ep_rew_means) / len(test_ep_rew_means)
			test_death_pct = summarize_deaths(test_death_stats)

			train_win_rate = compute_training_win_rate_from_events(run_dir)
			if train_win_rate is None:
				train_win_rate = float("nan")

			train_ep_rew_mean = compute_training_ep_rew_mean_from_events(run_dir)
			if train_ep_rew_mean is None:
				train_ep_rew_mean = float("nan")

			generalization_gap = train_win_rate - test_win_rate_mean

			writer.writerow(
				{
					"run_name": run_name,
					"env_name": env_name,
					"context": context_label,
					"train_win_rate": f"{train_win_rate:.4f}",
					"train_ep_rew_mean": f"{train_ep_rew_mean:.4f}",
					"train_death_saw_pct": f"{train_death_pct['saw']:.4f}",
					"train_death_enemy_pct": f"{train_death_pct['enemy']:.4f}",
					"train_death_lava_pct": f"{train_death_pct['lava']:.4f}",
					"train_death_timeout_pct": f"{train_death_pct['timeout']:.4f}",
					"train_death_unknown_pct": f"{train_death_pct['unknown']:.4f}",
					"test_win_rate_1": f"{test_win_rates[0]:.4f}",
					"test_win_rate_2": f"{test_win_rates[1]:.4f}",
					"test_win_rate_3": f"{test_win_rates[2]:.4f}",
					"test_win_rate_mean": f"{test_win_rate_mean:.4f}",
					"test_ep_rew_mean_1": f"{test_ep_rew_means[0]:.4f}",
					"test_ep_rew_mean_2": f"{test_ep_rew_means[1]:.4f}",
					"test_ep_rew_mean_3": f"{test_ep_rew_means[2]:.4f}",
					"test_ep_rew_mean": f"{test_ep_rew_mean:.4f}",
					"test_death_saw_pct": f"{test_death_pct['saw']:.4f}",
					"test_death_enemy_pct": f"{test_death_pct['enemy']:.4f}",
					"test_death_lava_pct": f"{test_death_pct['lava']:.4f}",
					"test_death_timeout_pct": f"{test_death_pct['timeout']:.4f}",
					"test_death_unknown_pct": f"{test_death_pct['unknown']:.4f}",
					"generalization_gap": f"{generalization_gap:.4f}",
				}
			)

	print(f"Saved results to: {output_path}")
	if args.print_table:
		print_table(output_path)


if __name__ == "__main__":
	main()
