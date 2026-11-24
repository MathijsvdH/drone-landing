import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

from envs.landing_aviary import LandingAviary
from evaluate import training_env_kwargs
from record_video import HOLD_SECONDS, caption, render


# Two models side by side on the same seeds, for results/compare/. The shorter episode freezes on its last frame.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record two models side by side on the same seeds.")
    parser.add_argument("--left", type=Path, required=True, help="model .zip shown on the left")
    parser.add_argument("--right", type=Path, required=True, help="model .zip shown on the right")
    parser.add_argument("--labels", nargs=2, required=True, metavar=("LEFT", "RIGHT"))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True, help="e.g. results/compare/<date>_<topic>/<name>.mp4")
    return parser.parse_args()


def episode(model_path: Path, label: str, seed: int) -> list[np.ndarray]:
    model = PPO.load(model_path, device="cpu")
    env = LandingAviary(**training_env_kwargs(model_path))
    obs, _ = env.reset(seed=seed)
    head = f"{label}   seed {seed}   pad {np.linalg.norm(env.pad_vel):.2f} m/s"
    frames = []
    while True:
        frames.append(caption(render(env), f"{head}   t = {env.step_counter / env.PYB_FREQ:4.1f} s"))
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
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
    env = LandingAviary()
    fps = env.CTRL_FREQ  # one frame per control step = real-time playback, as in record_video.py
    env.close()
    with imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8) as video:
        for seed in args.seeds:
            left = episode(args.left, args.labels[0], seed)
            right = episode(args.right, args.labels[1], seed)
            for i in range(max(len(left), len(right)) + int(HOLD_SECONDS * fps)):
                frame = np.hstack([half(left[min(i, len(left) - 1)]), half(right[min(i, len(right) - 1)])])
                video.append_data(np.ascontiguousarray(frame))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
