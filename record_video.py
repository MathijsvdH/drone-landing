import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pybullet as p
from PIL import Image, ImageDraw

from envs.landing_aviary import LandingAviary
from evaluate import checkpoint_label, load_agent, run_dir, training_env_kwargs

WIDTH, HEIGHT = 1280, 720
HOLD_SECONDS = 1.0  # freeze on the final frame so the touchdown is visible


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record an MP4 of a trained agent landing.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    parser.add_argument("--from-csv", type=Path, default=None,
                        help="pick seeds from an evaluate.py CSV by outcome instead of using --seeds")
    parser.add_argument("--outcomes", nargs="+", default=None,
                        help="with --from-csv: outcomes to show, in this order (default: all in the CSV, failures first)")
    parser.add_argument("--per-outcome", type=int, default=2, help="with --from-csv: episodes per outcome")
    parser.add_argument("--pad-speed", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="pad speed range in m/s; default: from the CSV if static, else the model's training range")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <run>/videos/<checkpoint>_seeds_<seeds>.mp4 (or <csv name>.mp4 with --from-csv)")
    return parser.parse_args()


def seeds_from_csv(csv: pd.DataFrame, outcomes: list[str] | None, per_outcome: int) -> dict[int, str]:
    # Failures first, so the interesting episodes are at the start of the video
    if outcomes is None:
        order = ["crashed", "out_of_bounds", "bad_landing", "timeout", "landed"]
        outcomes = [o for o in order if o in set(csv["outcome"])]
    picked = {}
    for outcome in outcomes:
        for seed in csv[csv["outcome"] == outcome]["seed"].head(per_outcome):
            picked[int(seed)] = outcome
    return picked


def replay_env_kwargs(args: argparse.Namespace, csv: pd.DataFrame | None) -> dict:
    # The pad speed per seed depends on the range it was sampled from, so a replay must use the
    # same range as the evaluation. A CSV with only static episodes means the range was (0, 0).
    # The observation always comes from the model's config.json.
    env_kwargs = training_env_kwargs(args.model)
    if args.pad_speed is not None:
        env_kwargs["pad_speed_range"] = tuple(args.pad_speed)
    elif csv is not None and "pad_speed" in csv and (csv["pad_speed"] == 0).all():
        env_kwargs["pad_speed_range"] = (0.0, 0.0)
    return env_kwargs


def video_env(**env_kwargs) -> LandingAviary:
    # The fast reset restores a saved world, which can't hold the extra floor tiles below; the full reset gives the
    # same episodes (checked bit-identical, 2026-11-19) and is fine for a handful of video episodes
    return LandingAviary(fast_reset=False, **env_kwargs)


def reset_with_floor(env: LandingAviary, seed: int):
    # The ground is 100 x 100 m visually; a 30 km/h pad drives ~125 m in an episode. Add a ring of visual-only copies
    # of the ground around it (collisions off, so the physics are untouched). The full reset removes them again.
    obs, info = env.reset(seed=seed)
    if env.pad_speed_range[1] > 2.0:
        for dx in (-100, 0, 100):
            for dy in (-100, 0, 100):
                if dx or dy:
                    tile = p.loadURDF("plane.urdf", [dx, dy, 0], useFixedBase=True, physicsClientId=env.CLIENT)
                    p.setCollisionFilterGroupMask(tile, -1, 0, 0, physicsClientId=env.CLIENT)
    return obs, info


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
    csv = pd.read_csv(args.from_csv) if args.from_csv else None
    if csv is not None:
        expected = seeds_from_csv(csv, args.outcomes, args.per_outcome)
        default_name = f"{args.from_csv.stem}.mp4"
    else:
        expected = {seed: None for seed in args.seeds}
        default_name = f"{checkpoint_label(args.model)}_seeds_{'_'.join(str(seed) for seed in args.seeds)}.mp4"
    out = args.out or run_dir(args.model) / "videos" / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    policy = load_agent(args.model, deterministic=True)
    env = video_env(**replay_env_kwargs(args, csv))
    fps = env.CTRL_FREQ  # one frame per control step = real-time playback

    with imageio.get_writer(out, fps=fps, codec="libx264", quality=8) as video:
        for seed, expected_outcome in expected.items():
            obs, _ = reset_with_floor(env, seed)
            pad = f"pad {np.linalg.norm(env.pad_vel):.2f} m/s"
            while True:
                t = env.step_counter / env.PYB_FREQ
                now = f"pad {np.linalg.norm(env.pad_vel):.2f} m/s"  # current speed: it changes in stage 3
                video.append_data(caption(render(env), f"seed {seed}   {now}   t = {t:4.1f} s"))
                obs, _, terminated, truncated, info = env.step(policy.act(obs))
                if terminated or truncated:
                    break
            result = info["outcome"]
            if info["touchdown_speed"] is not None:
                result += (f"   speed {info['touchdown_speed']:.2f} m/s"
                           f"   offset {info['touchdown_offset'] * 100:.1f} cm")
            final = caption(render(env), f"seed {seed}   {pad}   {result}")
            for _ in range(int(HOLD_SECONDS * fps)):
                video.append_data(final)
            print(f"seed {seed}: {result}")
            # Only happens if the replay differs from the evaluation (e.g. a --stochastic CSV or another pad speed)
            if expected_outcome is not None and info["outcome"] != expected_outcome:
                print(f"  WARNING: CSV says {expected_outcome}, replay gave {info['outcome']}")
    env.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
