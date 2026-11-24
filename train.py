import os

# Training runs on CPU, but PyTorch's Adam still queries CUDA on every step when a GPU is visible,
# so disconnecting the GPU for power saving mid-run crashed training. Hide it before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# The policy only sees batches of 8 observations, too small to split over threads. With the default (one thread per
# core) torch's idle helper threads busy-wait and took ~7 cores per run away from the simulation workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from envs.landing_aviary import (BAD_LANDING_PENALTY, LANDING_REWARD, MAX_TOUCHDOWN_OFFSET, MAX_TOUCHDOWN_SPEED,
                                 LandingAviary)

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
                    # Precision rings (any touchdown, whatever the limits), the same ones evaluation reports
                    for cm in (2, 3, 5):
                        self.logger.record_mean(f"touchdown/within_{cm}cm", float(info["touchdown_offset"] < cm / 100))
                    # Softness rings (run 8): how many touchdowns stay well under the speed limit
                    for speed in (0.15, 0.20, 0.25):
                        self.logger.record_mean(f"touchdown/below_{speed:.2f}ms", float(info["touchdown_speed"] < speed))
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


# Named recipes so a run is one short command. A preset only changes the defaults: any flag given on the command line
# still wins, and config.json records every resolved setting (and the preset name), so nothing is hidden.
# The library-style defaults below stay as they were, so the long commands in docs/ keep their meaning.
STAGE2 = {  # run 8: moving pad, safety margin, rate limiter, landing reward graded on offset and touchdown speed
    "timesteps": 5_000_000,
    "pad_speed": [0.2, 0.5],
    "own_velocity_obs": True,
    "train_touchdown_speed": 0.35,
    "train_touchdown_offset": 0.07,
    "action_rate_limit": 0.08,
    "landing_reward_edge": 50.0,
    "landing_speed_reward_edge": 50.0,
}
STAGE3 = {  # stage 2's recipe on a vehicle at 4-30 km/h that speeds up and brakes at 1-2 m/s^2 (2026-11-24)
    **STAGE2,
    "pad_speed": [4.0, 8.33],
    "speed_limit": 10.0,
    "drag": True,
    "start_following": True,
    "pad_per_physics_step": True,
    "pad_accel": [1.0, 2.0],
    "pad_accel_obs": True,
}
PRESETS = {"stage2": STAGE2, "stage3": STAGE3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent on LandingAviary.")
    parser.add_argument("--preset", choices=PRESETS, default=None,
                        help="named recipe (see PRESETS); flags given explicitly override it")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8, help="parallel environments (one process each)")
    parser.add_argument("--run-name", type=str, default=None, help="default: ppo_seed<seed>_<timestamp>")
    parser.add_argument("--pad-speed", type=float, nargs=2, default=[0.0, 0.0], metavar=("MIN", "MAX"),
                        help="pad speed range in m/s, sampled per episode with a random heading; 0 0 = static")
    parser.add_argument("--own-velocity-obs", action=argparse.BooleanOptionalAction, default=False,
                        help="add the drone's own velocity to the observation (12 -> 15 values)")
    # Safety margin: train against stricter landing limits than the milestone. Evaluation always uses the milestone.
    parser.add_argument("--train-touchdown-speed", type=float, default=MAX_TOUCHDOWN_SPEED,
                        help="max touchdown speed (m/s) that counts as landed during training")
    parser.add_argument("--train-touchdown-offset", type=float, default=MAX_TOUCHDOWN_OFFSET,
                        help="max distance to the pad centre (m) that counts as landed during training")
    parser.add_argument("--action-rate-limit", type=float, default=None,
                        help="max change of the velocity command per control step (m/s); adds 3 values to the observation")
    parser.add_argument("--landing-reward-edge", type=float, default=LANDING_REWARD,
                        help="landing reward at the edge of the allowed offset, linear up to the full reward at the "
                             "centre (precision); default = flat reward")
    parser.add_argument("--landing-speed-reward-edge", type=float, default=LANDING_REWARD,
                        help="landing reward at the touchdown speed limit, linear up to the full reward at 0 m/s "
                             "(softness); default = no speed term")
    parser.add_argument("--bad-landing-penalty", type=float, default=BAD_LANDING_PENALTY,
                        help="reward for touching the pad too fast or off-centre (a miss)")
    # Fast pad / changing speed (all off by default)
    parser.add_argument("--speed-limit", type=float, default=1.0,
                        help="max commanded speed (m/s); must exceed the pad speed (e.g. 10 for a 30 km/h pad)")
    parser.add_argument("--drag", action=argparse.BooleanOptionalAction, default=False, help="air drag on (Physics.PYB_DRAG)")
    parser.add_argument("--start-following", action=argparse.BooleanOptionalAction, default=False,
                        help="drone starts flying along with the pad (+- 0.5 m/s) instead of hovering")
    parser.add_argument("--pad-per-physics-step", action=argparse.BooleanOptionalAction, default=False,
                        help="move the pad every physics step instead of every control step (needed for fast pads)")
    parser.add_argument("--pad-accel", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="pad changes speed within --pad-speed, accelerating/braking at this rate (m/s^2)")
    parser.add_argument("--pad-accel-obs", action=argparse.BooleanOptionalAction, default=False,
                        help="the pad broadcasts its acceleration: 3 values appended to the observation (stage 3)")
    parser.add_argument("--relative-action-scale", type=float, default=None,
                        help="actions are a correction (max this many m/s) on top of the pad's velocity instead of an "
                             "absolute velocity (stage 3, residual control); default = absolute")
    parser.add_argument("--normalize-obs", action=argparse.BooleanOptionalAction, default=False,
                        help="scale each observation value by its running mean / std (VecNormalize); the statistics "
                             "are saved next to every checkpoint and used by evaluation")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000, help="in total timesteps")
    # PPO hyperparameters, SB3 defaults
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048, help="rollout length per env")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    preset = parser.parse_known_args()[0].preset
    if preset:
        parser.set_defaults(**PRESETS[preset])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"ppo_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path("results") / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    env_kwargs = {"pad_speed_range": tuple(args.pad_speed),
                  "own_velocity_obs": args.own_velocity_obs,
                  "touchdown_speed_limit": args.train_touchdown_speed,
                  "touchdown_offset_limit": args.train_touchdown_offset,
                  "action_rate_limit": args.action_rate_limit,
                  "landing_reward_edge": args.landing_reward_edge,
                  "landing_speed_reward_edge": args.landing_speed_reward_edge,
                  "bad_landing_penalty": args.bad_landing_penalty,
                  "speed_limit": args.speed_limit,
                  "drag": args.drag,
                  "start_following": args.start_following,
                  "pad_per_physics_step": args.pad_per_physics_step,
                  "pad_accel_range": tuple(args.pad_accel) if args.pad_accel else None,
                  "pad_accel_obs": args.pad_accel_obs,
                  "relative_action_scale": args.relative_action_scale}
    # env_kwargs and the commit are a record of exactly what was trained; evaluate.py reads the flat keys
    config = {**vars(args), "env_kwargs": env_kwargs, "git_commit": git_commit(),
              "omp_num_threads": os.environ["OMP_NUM_THREADS"]}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Each env gets seed + i, so the whole run is reproducible from --seed
    env = make_vec_env(LandingAviary,
                       n_envs=args.n_envs,
                       seed=args.seed,
                       env_kwargs=env_kwargs,
                       vec_env_cls=SubprocVecEnv,
                       monitor_dir=str(run_dir / "monitor"),
                       monitor_kwargs={"info_keywords": ("outcome", "touchdown_speed", "touchdown_offset")})
    if args.normalize_obs:
        # Stage 3: own velocity reaches ~8 m/s while the relative position that decides a landing is ~0.3 m, so the
        # raw inputs differ ~15x in scale. Rewards stay raw: the reward is the thing we compare between runs.
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=args.gamma)

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
                                     name_prefix="model",
                                     save_vecnormalize=True)  # only does something with --normalize-obs
    model.learn(total_timesteps=args.timesteps,
                callback=CallbackList([EpisodeStatsCallback(), checkpoints, StopAtTimesteps(args.timesteps)]),
                tb_log_name="tb")
    model.save(run_dir / "model")
    if args.normalize_obs:
        env.save(str(run_dir / "vecnormalize.pkl"))
    env.close()
    print(f"Saved model to {run_dir / 'model.zip'}")


if __name__ == "__main__":
    main()
