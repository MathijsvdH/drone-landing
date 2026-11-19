import os

# Training runs on CPU, but PyTorch's Adam still queries CUDA on every step when a GPU is visible,
# so disconnecting the GPU for power saving mid-run crashed training. Hide it before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.landing_aviary import MAX_TOUCHDOWN_OFFSET, MAX_TOUCHDOWN_SPEED, LandingAviary

OUTCOMES = ("landed", "bad_landing", "crashed", "out_of_bounds", "timeout")


# Logs per-episode reward terms, outcome rates and touchdown metrics to TensorBoard
class EpisodeStatsCallback(BaseCallback):
    def _on_training_start(self) -> None:
        self.sums = [defaultdict(float) for _ in range(self.training_env.num_envs)]

    def _on_step(self) -> bool:
        for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"])):
            for key, value in info.items():
                if key.startswith("reward/"):
                    self.sums[i][key] += value
            if done:
                for key, total in self.sums[i].items():
                    self.logger.record_mean(f"episode_{key}", total)
                for outcome in OUTCOMES:
                    self.logger.record_mean(f"outcome/{outcome}", float(info["outcome"] == outcome))
                # Why landings fail, against the training limits (stricter than the milestone with a safety margin)
                bad = info["outcome"] == "bad_landing"
                self.logger.record_mean("bad_landing/too_fast", float(bad and info["too_fast"]))
                self.logger.record_mean("bad_landing/off_centre", float(bad and info["off_centre"]))
                if info["outcome"] in ("landed", "bad_landing"):
                    self.logger.record_mean("touchdown/speed", info["touchdown_speed"])
                    self.logger.record_mean("touchdown/offset", info["touchdown_offset"])
                # Success under the milestone criteria that evaluation uses. Equals outcome/landed without a
                # safety margin, so runs with and without one can be compared on the same curve.
                self.logger.record_mean("outcome/landed_milestone", float(
                    info["outcome"] in ("landed", "bad_landing")
                    and info["touchdown_speed"] < MAX_TOUCHDOWN_SPEED
                    and info["touchdown_offset"] < MAX_TOUCHDOWN_OFFSET))
                self.sums[i].clear()
        return True


# PPO only checks the timestep budget between rollouts, so --timesteps 5000000 used to run to 5,013,504 and do
# one more update after the 5M checkpoint. Stopping here instead makes model.zip the same as the last checkpoint.
class StopAtTimesteps(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        return self.num_timesteps < self.total_timesteps


def git_commit() -> str:
    # Which code a run was trained with; "-dirty" if there were uncommitted changes
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent on LandingAviary.")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8, help="parallel environments (one process each)")
    parser.add_argument("--run-name", type=str, default=None, help="default: ppo_seed<seed>_<timestamp>")
    parser.add_argument("--pad-speed", type=float, nargs=2, default=[0.0, 0.0], metavar=("MIN", "MAX"),
                        help="pad speed range in m/s, sampled per episode with a random heading; 0 0 = static")
    parser.add_argument("--own-velocity-obs", action="store_true",
                        help="add the drone's own velocity to the observation (12 -> 15 values)")
    # Safety margin: train against stricter landing limits than the milestone. Evaluation always uses the milestone.
    parser.add_argument("--train-touchdown-speed", type=float, default=MAX_TOUCHDOWN_SPEED,
                        help="max touchdown speed (m/s) that counts as landed during training")
    parser.add_argument("--train-touchdown-offset", type=float, default=MAX_TOUCHDOWN_OFFSET,
                        help="max distance to the pad centre (m) that counts as landed during training")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000, help="in total timesteps")
    # PPO hyperparameters, SB3 defaults
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048, help="rollout length per env")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"ppo_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path("results") / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    env_kwargs = {"pad_speed_range": tuple(args.pad_speed),
                  "own_velocity_obs": args.own_velocity_obs,
                  "touchdown_speed_limit": args.train_touchdown_speed,
                  "touchdown_offset_limit": args.train_touchdown_offset}
    # env_kwargs and the commit are a record of exactly what was trained; evaluate.py reads the flat keys
    config = {**vars(args), "env_kwargs": env_kwargs, "git_commit": git_commit()}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Each env gets seed + i, so the whole run is reproducible from --seed
    env = make_vec_env(LandingAviary,
                       n_envs=args.n_envs,
                       seed=args.seed,
                       env_kwargs=env_kwargs,
                       vec_env_cls=SubprocVecEnv,
                       monitor_dir=str(run_dir / "monitor"),
                       monitor_kwargs={"info_keywords": ("outcome", "touchdown_speed", "touchdown_offset")})

    model = PPO("MlpPolicy", env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                seed=args.seed,
                device="cpu",  # small MLP policy: CPU is faster than GPU transfer overhead
                tensorboard_log=str(run_dir),
                verbose=1)

    checkpoints = CheckpointCallback(save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                                     save_path=str(run_dir / "checkpoints"),
                                     name_prefix="model")
    model.learn(total_timesteps=args.timesteps,
                callback=CallbackList([EpisodeStatsCallback(), checkpoints, StopAtTimesteps(args.timesteps)]),
                tb_log_name="tb")
    model.save(run_dir / "model")
    env.close()
    print(f"Saved model to {run_dir / 'model.zip'}")


if __name__ == "__main__":
    main()
