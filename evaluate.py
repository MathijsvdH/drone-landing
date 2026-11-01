import argparse
import time
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from gym_pybullet_drones.utils.utils import sync

from envs.landing_aviary import LandingAviary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-seed evaluation episodes and write a CSV.")
    parser.add_argument("--model", type=Path, required=True, help="path to a saved model .zip")
    parser.add_argument("--episodes", type=int, default=100)
    # Episode i uses seed + i. Default 1000 keeps eval starts apart from the training seeds (0..n_envs-1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None, help="default: eval_seed<seed>.csv next to the model")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the mean")
    parser.add_argument("--gui", action="store_true", help="watch in real time (demo only)")
    return parser.parse_args()


def run_episode(env: LandingAviary, model: PPO, seed: int, deterministic: bool, gui: bool) -> dict:
    obs, _ = env.reset(seed=seed)
    start = env.INIT_XYZS[0].copy()
    reward_sums: dict[str, float] = {}
    total, steps, wall_start = 0.0, 0, time.time()
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
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
        **reward_sums,
    }


def main() -> None:
    args = parse_args()
    out = args.out or args.model.parent / f"eval_seed{args.seed}.csv"
    model = PPO.load(args.model, device="cpu")
    env = LandingAviary(gui=args.gui)

    rows = [run_episode(env, model, args.seed + i, not args.stochastic, args.gui) for i in range(args.episodes)]
    env.close()

    df = pd.DataFrame(rows)
    df.insert(0, "policy", "agent")
    df.insert(1, "model", str(args.model))
    df.to_csv(out, index=False)

    touched = df[df["outcome"].isin(["landed", "bad_landing"])]
    print(f"\n{len(df)} episodes, seeds {args.seed}..{args.seed + args.episodes - 1}")
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
