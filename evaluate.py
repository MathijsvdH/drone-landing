import os

# One torch thread: the policy sees one observation at a time, and parallel workers would otherwise fight over cores
# with busy-waiting helper threads (same as train.py; evaluation results are identical either way)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import multiprocessing
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from gym_pybullet_drones.utils.utils import sync

from envs.landing_aviary import LandingAviary
from pid_baseline import STAGE3_GAINS, PIDLander


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-seed evaluation episodes and write a CSV.")
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--model", type=Path, help="path to a saved model .zip")
    policy.add_argument("--pid", action="store_true", help="evaluate the PID baseline instead of an agent")
    parser.add_argument("--episodes", type=int, default=100)
    # Episode i uses seed + i. Default 1000 keeps eval starts apart from the training seeds (0..n_envs-1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <run>/eval/<checkpoint>_seed<seed>_n<episodes>[_<settings>].csv "
                             "(results/pid/eval/ for --pid)")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel processes; each runs a block of seeds, so the CSV is the same as with 1")
    parser.add_argument("--pad-speed", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="pad speed range in m/s; default: the model's training range (static for --pid)")
    parser.add_argument("--own-velocity-obs", action="store_true",
                        help="PID only: 15-value observation and pad-velocity feedforward "
                             "(agents always use the observation they were trained with)")
    parser.add_argument("--action-rate-limit", type=float, default=None,
                        help="PID only: max change of the velocity command per step (m/s), the same safety layer "
                             "the agent was trained with (agents use the limit from their config.json)")
    parser.add_argument("--speed-limit", type=float, default=None,
                        help="PID only: max commanded speed (m/s), default 1.0 (agents use their config.json)")
    # Task settings: default from an agent's config.json (off for the PID); given here, they override it
    parser.add_argument("--drag", action="store_true", help="air drag on")
    parser.add_argument("--start-following", action="store_true", help="drone starts flying along with the pad")
    parser.add_argument("--pad-per-physics-step", action="store_true", help="move the pad every physics step")
    parser.add_argument("--pad-accel", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="pad changes speed, accelerating/braking at this rate (m/s^2)")
    parser.add_argument("--pid-stage3", action="store_true",
                        help="PID only: the gains tuned for stage 3 (pid_baseline.STAGE3_GAINS) instead of stage 2")
    parser.add_argument("--pad-accel-obs", action="store_true",
                        help="PID only: the pad's acceleration in the observation (agents use their config.json)")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the mean")
    parser.add_argument("--gui", action="store_true", help="watch in real time (demo only)")
    args = parser.parse_args()
    if args.own_velocity_obs and not args.pid:
        parser.error("--own-velocity-obs is only for --pid; an agent's observation comes from its config.json")
    if args.action_rate_limit is not None and not args.pid:
        parser.error("--action-rate-limit is only for --pid; an agent's limit comes from its config.json")
    if args.speed_limit is not None and not args.pid:
        parser.error("--speed-limit is only for --pid; it changes what an agent's actions mean")
    if args.pid_stage3 and not args.pid:
        parser.error("--pid-stage3 is only for --pid")
    if args.pad_accel_obs and not args.pid:
        parser.error("--pad-accel-obs is only for --pid; an agent's observation comes from its config.json")
    if args.gui and args.workers > 1:
        parser.error("--gui needs --workers 1")
    return args


class AgentPolicy:
    # Same reset/act interface as PIDLander, so both run through the exact same episode loop

    def __init__(self, model: PPO, deterministic: bool, normalizer=None):
        self.model = model
        self.deterministic = deterministic
        self.normalizer = normalizer  # the training run's VecNormalize statistics (frozen), or None

    def reset(self) -> None:
        pass

    def act(self, obs):
        if self.normalizer is not None:
            obs = self.normalizer.normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action


def run_dir(model: Path) -> Path:
    # Checkpoints live in <run>/checkpoints/, the final model in <run>/
    return model.parent.parent if model.parent.name == "checkpoints" else model.parent


def checkpoint_label(model: Path) -> str:
    # model_3000000_steps.zip -> 3000k, model.zip -> final
    if model.stem.startswith("model_") and model.stem.endswith("_steps"):
        return f"{int(model.stem.split('_')[1]) // 1000}k"
    return "final" if model.stem == "model" else model.stem


def training_config(model: Path) -> dict:
    # Checkpoints live in <run>/checkpoints/, so look one folder up as well
    config_path = next(path for path in (model.parent / "config.json", model.parent.parent / "config.json")
                       if path.exists())
    return json.loads(config_path.read_text())


def training_env_kwargs(model: Path) -> dict:
    # The env settings a model was trained with, so it is evaluated on the same task and observation
    config = training_config(model)
    # Older runs have no pad_speed (static pad) or own_velocity_obs (12-value observation) in their config.
    # Training landing limits (safety margin) and reward settings are left out on purpose: evaluation always uses the
    # milestone limits, and rewards don't change the outcome (only the CSV's return column).
    return {
        "pad_speed_range": tuple(config.get("pad_speed", (0.0, 0.0))),
        "own_velocity_obs": config.get("own_velocity_obs", False),
        "action_rate_limit": config.get("action_rate_limit"),  # None for runs before run 6
        # Fast pad / changing speed (runs before 7 don't have these: slow pad, hover start)
        "speed_limit": config.get("speed_limit", 1.0),
        "drag": config.get("drag", False),
        "start_following": config.get("start_following", False),
        "pad_per_physics_step": config.get("pad_per_physics_step", False),
        "pad_accel_range": tuple(config["pad_accel"]) if config.get("pad_accel") else None,
        "pad_accel_obs": config.get("pad_accel_obs", False),
        "relative_action_scale": config.get("relative_action_scale"),  # None = absolute actions (before stage 3b)
    }


def run_episode(env: LandingAviary, policy: AgentPolicy | PIDLander, seed: int, gui: bool) -> dict:
    obs, _ = env.reset(seed=seed)
    policy.reset()
    start = env.INIT_XYZS[0].copy()
    pad_vel = env.pad_vel.copy()
    reward_sums: dict[str, float] = {}
    total, steps, wall_start = 0.0, 0, time.time()
    max_speed, max_tilt = 0.0, 0.0  # in-flight safety: loss of control shows here even when the landing still counts
    while True:
        obs, reward, terminated, truncated, info = env.step(policy.act(obs))
        total += reward
        steps += 1
        state = env._getDroneStateVector(0)
        max_speed = max(max_speed, float(np.linalg.norm(state[10:13])))
        max_tilt = max(max_tilt, float(np.degrees(max(abs(state[7]), abs(state[8])))))
        for key, value in info.items():
            if key.startswith("reward/"):
                reward_sums[key] = reward_sums.get(key, 0.0) + value
        if gui:
            sync(steps, wall_start, env.CTRL_TIMESTEP)
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "outcome": info["outcome"],
        "success": info["outcome"] == "landed",
        "steps": steps,
        "duration_s": steps * env.CTRL_TIMESTEP,
        "return": total,
        "touchdown_speed": info["touchdown_speed"],
        "touchdown_offset": info["touchdown_offset"],
        "max_speed": max_speed,
        "max_tilt_deg": max_tilt,
        "start_x": start[0],
        "start_y": start[1],
        "start_z": start[2],
        "pad_speed": float(np.linalg.norm(pad_vel)),
        "pad_heading_deg": float(np.degrees(np.arctan2(pad_vel[1], pad_vel[0]))),
        **({"pad_accel": env.pad_accel, "pad_speed_end": float(np.linalg.norm(env.pad_vel))}
           if env.pad_accel_range is not None else {}),
        **reward_sums,
    }


def load_agent(model: Path, deterministic: bool) -> AgentPolicy:
    # A model trained with --normalize-obs only works with the statistics saved at the same step:
    # checkpoints/model_<steps>_steps.zip -> checkpoints/model_vecnormalize_<steps>_steps.pkl, model.zip -> vecnormalize.pkl
    normalizer = None
    if training_config(model).get("normalize_obs"):
        stats = (model.with_name(model.name.replace("model_", "model_vecnormalize_")).with_suffix(".pkl")
                 if model.parent.name == "checkpoints" else model.parent / "vecnormalize.pkl")
        with open(stats, "rb") as file:
            normalizer = pickle.load(file)
        normalizer.training = False  # frozen: evaluation must not change the statistics
    return AgentPolicy(PPO.load(model, device="cpu"), deterministic, normalizer)


def make_policy(args: argparse.Namespace, env: LandingAviary) -> AgentPolicy | PIDLander:
    if args.pid:
        gains = STAGE3_GAINS if args.pid_stage3 else {}
        return PIDLander(speed_limit=env.SPEED_LIMIT, dt=env.CTRL_TIMESTEP, feedforward=args.own_velocity_obs, **gains)
    return load_agent(args.model, deterministic=not args.stochastic)


def run_seeds(job: tuple) -> tuple[list[dict], int]:
    # One worker: its own env and policy, a contiguous block of seeds. Every episode starts from env.reset(seed),
    # so the result of a seed does not depend on which episodes ran before it in the same env.
    args, env_kwargs, seeds = job
    env = LandingAviary(gui=args.gui, **env_kwargs)
    policy = make_policy(args, env)
    rows = [run_episode(env, policy, int(seed), args.gui) for seed in seeds]
    obs_size = env.obs_size
    env.close()
    return rows, obs_size


def default_out(args: argparse.Namespace, pad_speed: tuple) -> Path:
    # <policy>_seed<first>_n<episodes>[_<task settings>].csv; task settings only when they are not the policy's own
    if args.pid:
        folder = Path("results/pid/eval")
        name = "pid" + ("_ff" if args.own_velocity_obs else "")
        name += f"_rate{args.action_rate_limit:g}" if args.action_rate_limit is not None else ""
        name += f"_vmax{args.speed_limit:g}" if args.speed_limit is not None else ""
        name += "_accelobs" if args.pad_accel_obs else ""
        name += "_stage3gains" if args.pid_stage3 else ""
    else:
        folder = run_dir(args.model) / "eval"
        name = checkpoint_label(args.model) + ("_stochastic" if args.stochastic else "")
    name += f"_seed{args.seed}_n{args.episodes}"
    if args.pid or args.pad_speed is not None:  # the PID has no task of its own, so its pad speed is always named
        name += f"_pad{pad_speed[0]:g}-{pad_speed[1]:g}"
    name += "".join(f"_{label}" for label, flag in (("drag", args.drag), ("follow", args.start_following),
                                                    ("pps", args.pad_per_physics_step)) if flag)
    if args.pad_accel is not None:
        name += f"_accel{args.pad_accel[0]:g}-{args.pad_accel[1]:g}"
    return folder / f"{name}.csv"


def main() -> None:
    args = parse_args()
    if args.pid:
        env_kwargs = {"pad_speed_range": (0.0, 0.0), "own_velocity_obs": args.own_velocity_obs,
                      "action_rate_limit": args.action_rate_limit,
                      "speed_limit": args.speed_limit if args.speed_limit is not None else 1.0,
                      "pad_accel_obs": args.pad_accel_obs}
    else:
        env_kwargs = training_env_kwargs(args.model)
    if args.pad_speed is not None:
        env_kwargs["pad_speed_range"] = tuple(args.pad_speed)
    for flag in ("drag", "start_following", "pad_per_physics_step"):
        if getattr(args, flag):
            env_kwargs[flag] = True
    if args.pad_accel is not None:
        env_kwargs["pad_accel_range"] = tuple(args.pad_accel)
    pad_speed = env_kwargs["pad_speed_range"]
    out = args.out or default_out(args, pad_speed)
    out.parent.mkdir(parents=True, exist_ok=True)

    blocks = np.array_split(np.arange(args.seed, args.seed + args.episodes), max(1, min(args.workers, args.episodes)))
    jobs = [(args, env_kwargs, block) for block in blocks]
    if len(jobs) == 1:
        results = [run_seeds(jobs[0])]
    else:
        # spawn: fresh processes, no forked torch / PyBullet state from this one
        with multiprocessing.get_context("spawn").Pool(len(jobs)) as pool:
            results = pool.map(run_seeds, jobs)
    rows = [row for block_rows, _ in results for row in block_rows]
    obs_size = results[0][1]

    df = pd.DataFrame(rows)
    df.insert(0, "policy", "pid" if args.pid else "agent")
    df.insert(1, "model", "" if args.pid else str(args.model))
    df.to_csv(out, index=False)

    touched = df[df["outcome"].isin(["landed", "bad_landing"])]
    print(f"\n{len(df)} episodes, seeds {args.seed}..{args.seed + args.episodes - 1}, "
          f"pad speed {pad_speed[0]:g}-{pad_speed[1]:g} m/s, {obs_size}-value observation")
    print(f"success rate:      {df['success'].mean():.0%}")
    print(f"outcomes:          {df['outcome'].value_counts().to_dict()}")
    if len(touched):
        print(f"touchdown speed:   median {touched['touchdown_speed'].median():.3f} m/s, "
              f"max {touched['touchdown_speed'].max():.3f}")
        print(f"touchdown offset:  median {touched['touchdown_offset'].median():.3f} m, "
              f"max {touched['touchdown_offset'].max():.3f}")
    if len(touched):
        rings = ", ".join(f"{(touched['touchdown_offset'] < cm / 100).mean():.0%} < {cm} cm" for cm in (2, 3, 5))
        print(f"precision:         {rings}")
        soft = ", ".join(f"{(touched['touchdown_speed'] < v).mean():.0%} < {v:.2f}" for v in (0.15, 0.20, 0.25))
        print(f"softness:          {soft} m/s")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
