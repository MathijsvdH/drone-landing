import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from envs.landing_aviary import LandingAviary

WIDTH, HEIGHT = 1280, 720
HOLD_SECONDS = 1.0  # freeze on the final frame so the touchdown is visible


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record an MP4 of a trained agent landing.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    parser.add_argument("--out", type=Path, default=None, help="default: landing.mp4 next to the model")
    return parser.parse_args()


def render(env: LandingAviary) -> np.ndarray:
    # Camera looks at the point between drone and pad, so both stay in view
    drone = env._getDroneStateVector(0)[0:3]
    target = (drone + env.pad_pos) / 2
    distance = 1.2 + 0.8 * np.linalg.norm(drone - env.pad_pos)
    view = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=target, distance=distance,
                                               yaw=-35, pitch=-25, roll=0, upAxisIndex=2,
                                               physicsClientId=env.CLIENT)
    proj = p.computeProjectionMatrixFOV(fov=50, aspect=WIDTH / HEIGHT, nearVal=0.05, farVal=50,
                                        physicsClientId=env.CLIENT)
    _, _, rgba, _, _ = p.getCameraImage(WIDTH, HEIGHT, view, proj, shadow=1,
                                        renderer=p.ER_TINY_RENDERER, physicsClientId=env.CLIENT)
    return np.reshape(rgba, (HEIGHT, WIDTH, 4))[:, :, :3].astype(np.uint8)


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 44), fill=(20, 20, 20))
    draw.text((16, 12), text, fill=(240, 240, 240), font_size=22)
    return np.asarray(image)


def main() -> None:
    args = parse_args()
    out = args.out or args.model.parent / "landing.mp4"
    model = PPO.load(args.model, device="cpu")
    env = LandingAviary()
    fps = env.CTRL_FREQ  # one frame per control step = real-time playback

    with imageio.get_writer(out, fps=fps, codec="libx264", quality=8) as video:
        for seed in args.seeds:
            obs, _ = env.reset(seed=seed)
            while True:
                t = env.step_counter / env.PYB_FREQ
                video.append_data(caption(render(env), f"seed {seed}   t = {t:4.1f} s"))
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            result = info["outcome"]
            if info["touchdown_speed"] is not None:
                result += (f"   speed {info['touchdown_speed']:.2f} m/s"
                           f"   offset {info['touchdown_offset'] * 100:.1f} cm")
            final = caption(render(env), f"seed {seed}   {result}")
            for _ in range(int(HOLD_SECONDS * fps)):
                video.append_data(final)
            print(f"seed {seed}: {result}")
    env.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
