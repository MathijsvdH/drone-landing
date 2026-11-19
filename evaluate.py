import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from gym_pybullet_drones.utils.utils import sync

from envs.landing_aviary import LandingAviary
from pid_baseline import PIDLander


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-seed evaluation episodes and write a CSV.")
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--model", type=Path, help="path to a saved model .zip")
    policy.add_argument("--pid", action="store_true", help="evaluate the PID baseline instead of an agent")
    parser.add_argument("--episodes", type=int, default=100)
    # Episode i uses seed + i. Default 1000 keeps eval starts apart from the training seeds (0..n_envs-1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: eval_seed<seed>.csv next to the model, or in results/pid/ for --pid")
    parser.add_argument("--pad-speed", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="pad speed range in m/s; default: the model's training range (static for --pid)")
    parser.add_argument("--own-velocity-obs", action="store_true",
                        help="PID only: 15-value observation and pad-velocity feedforward "
                             "(agents always use the observation they were trained with)")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the mean")
    parser.add_argument("--gui", action="store_true", help="watch in real time (demo only)")
    args = parser.parse_args()
    if args.own_velocity_obs and not args.pid:
        parser.error("--own-velocity-obs is only for --pid; an agent's observation comes from its config.json")
    return args


class AgentPolicy:
    # Same reset/act interface as PIDLander, so both run through the exact same episode loop

    def __init__(self, model: PPO, deterministic: bool):
        self.model = model
        self.deterministic = deterministic

    def reset(self) -> None:
        pass

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action


def training_env_kwargs(model: Path) -> dict:
    # The env settings a model was trained with, so it is evaluated on the same task and observation.
    # Checkpoints live in <run>/checkpoints/, so look one folder up as well.
    config_path = next(path for path in (model.parent / "config.json", model.parent.parent / "config.json")
                       if path.exists())
    config = json.loads(config_path.read_text())
    # Older runs have no pad_speed (static pad) or own_velocity_obs (12-value observation) in their config.
    # Training landing limits (safety margin) are left out on purpose: evaluation always uses the milestone limits.
    return {
        "pad_speed_range": tuple(config.get("pad_speed", (0.0, 0.0))),
        "own_velocity_obs": config.get("own_velocity_obs", False),
    }


def run_episode(env: LandingAviary, policy: AgentPolicy | PIDLander, seed: int, gui: bool) -> dict:
    obs, _ = env.reset(seed=seed)
    policy.reset()
    start = env.INIT_XYZS[0].copy()
    pad_vel = env.pad_vel.copy()
    reward_sums: dict[str, float] = {}
    total, steps, wall_start = 0.0, 0, time.time()
    while True:
        obs, reward, terminated, truncated, info = env.step(policy.act(obs))
        total += reward
        steps += 1
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
        "start_x": start[0],
        "start_y": start[1],
        "start_z": start[2],
        "pad_speed": float(np.linalg.norm(pad_vel)),
        "pad_heading_deg": float(np.degrees(np.arctan2(pad_vel[1], pad_vel[0]))),
        **reward_sums,
    }


def main() -> None:
    args = parse_args()
    if args.pid:
        env_kwargs = {"pad_speed_range": (0.0, 0.0), "own_velocity_obs": args.own_velocity_obs}
    else:
        env_kwargs = training_env_kwargs(args.model)
    if args.pad_speed is not None:
        env_kwargs["pad_speed_range"] = tuple(args.pad_speed)
    pad_speed = env_kwargs["pad_speed_range"]
    env = LandingAviary(gui=args.gui, **env_kwargs)
    if args.pid:
        out_dir = Path("results/pid")
        policy = PIDLander(speed_limit=env.SPEED_LIMIT, dt=env.CTRL_TIMESTEP, feedforward=args.own_velocity_obs)
    else:
        out_dir = args.model.parent
        policy = AgentPolicy(PPO.load(args.model, device="cpu"), deterministic=not args.stochastic)
    # Static-pad files keep their old name; moving-pad files get the speed range in the name
    suffix = f"_pad{pad_speed[0]:g}-{pad_speed[1]:g}" if pad_speed[1] > 0 else ""
    if args.pid and args.own_velocity_obs:
        suffix += "_feedforward"  # don't overwrite the plain PID baseline's CSV
    out = args.out or out_dir / f"eval_seed{args.seed}{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [run_episode(env, policy, args.seed + i, args.gui) for i in range(args.episodes)]
    env.close()

    df = pd.DataFrame(rows)
    df.insert(0, "policy", "pid" if args.pid else "agent")
    df.insert(1, "model", "" if args.pid else str(args.model))
    df.to_csv(out, index=False)

    touched = df[df["outcome"].isin(["landed", "bad_landing"])]
    print(f"\n{len(df)} episodes, seeds {args.seed}..{args.seed + args.episodes - 1}, "
          f"pad speed {pad_speed[0]:g}-{pad_speed[1]:g} m/s, {env.obs_size}-value observation")
    print(f"success rate:      {df['success'].mean():.0%}")
    print(f"outcomes:          {df['outcome'].value_counts().to_dict()}")
    if len(touched):
        print(f"touchdown speed:   median {touched['touchdown_speed'].median():.3f} m/s, "
              f"max {touched['touchdown_speed'].max():.3f}")
        print(f"touchdown offset:  median {touched['touchdown_offset'].median():.3f} m, "
              f"max {touched['touchdown_offset'].max():.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
