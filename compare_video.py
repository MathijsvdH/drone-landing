import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

from envs.landing_aviary import LandingAviary
from evaluate import AgentPolicy, training_env_kwargs
from pid_baseline import STAGE3_GAINS, PIDLander
from record_video import HOLD_SECONDS, caption, render, reset_with_floor, video_env


# Two models side by side on the same seeds, for results/compare/. The shorter episode freezes on its last frame.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record two models side by side on the same seeds.")
    parser.add_argument("--left", type=Path, required=True, help="model .zip shown on the left")
    parser.add_argument("--right", type=Path, required=True,
                        help="model .zip shown on the right, or 'pid': the PID baseline on the left model's task "
                             "(with feedforward, and the stage-3 gains when the pad changes speed)")
    parser.add_argument("--labels", nargs=2, required=True, metavar=("LEFT", "RIGHT"))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True, help="e.g. results/compare/<date>_<topic>/<name>.mp4")
    return parser.parse_args()


def make_policy(path: Path, env: LandingAviary) -> AgentPolicy | PIDLander:
    if str(path) == "pid":
        gains = STAGE3_GAINS if env.pad_accel_range is not None else {}
        return PIDLander(speed_limit=env.SPEED_LIMIT, dt=env.CTRL_TIMESTEP, feedforward=env.own_velocity_obs, **gains)
    return AgentPolicy(PPO.load(path, device="cpu"), deterministic=True)


def episode(path: Path, env_kwargs: dict, label: str, seed: int) -> list[np.ndarray]:
    env = video_env(**env_kwargs)
    policy = make_policy(path, env)
    obs, _ = reset_with_floor(env, seed)
    policy.reset()
    frames = []
    while True:
        now = f"pad {np.linalg.norm(env.pad_vel):.2f} m/s"  # current speed: it changes in stage 3
        frames.append(caption(render(env), f"{label}   seed {seed}   {now}   t = {env.step_counter / env.PYB_FREQ:4.1f} s"))
        obs, _, terminated, truncated, info = env.step(policy.act(obs))
        if terminated or truncated:
            break
    result = info["outcome"]
    if info["touchdown_speed"] is not None:
        result += f"  {info['touchdown_speed']:.2f} m/s  {info['touchdown_offset'] * 100:.1f} cm"
    frames.append(caption(render(env), f"{label}   seed {seed}   {result}"))
    env.close()
    print(f"{label} seed {seed}: {result}")
    return frames


def half(frame: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).resize((frame.shape[1] // 2, frame.shape[0] // 2)))


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Each side gets the settings it was trained with (its own observation); the task itself must be the same.
    # The PID flies the left model's task with the left model's observation.
    left_kwargs = training_env_kwargs(args.left)
    # The PID gives absolute velocity commands with its own feedforward, so it never uses relative actions
    right_kwargs = ({**left_kwargs, "relative_action_scale": None} if str(args.right) == "pid"
                    else training_env_kwargs(args.right))
    interface_keys = {"own_velocity_obs", "pad_accel_obs", "relative_action_scale"}  # what a policy sees and outputs
    if {k: v for k, v in left_kwargs.items() if k not in interface_keys} != \
            {k: v for k, v in right_kwargs.items() if k not in interface_keys}:
        raise SystemExit("left and right were trained on different tasks")
    env = LandingAviary()
    fps = env.CTRL_FREQ  # one frame per control step = real-time playback, as in record_video.py
    env.close()
    with imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8) as video:
        for seed in args.seeds:
            left = episode(args.left, left_kwargs, args.labels[0], seed)
            right = episode(args.right, right_kwargs, args.labels[1], seed)
            for i in range(max(len(left), len(right)) + int(HOLD_SECONDS * fps)):
                frame = np.hstack([half(left[min(i, len(left) - 1)]), half(right[min(i, len(right) - 1)])])
                video.append_data(np.ascontiguousarray(frame))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
