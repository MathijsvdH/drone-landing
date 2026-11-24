import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import itertools
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from envs.landing_aviary import LandingAviary
from evaluate import run_episode
from pid_baseline import PIDLander

# Tunes the PID for stage 3 (fast vehicle that speeds up and brakes) on validation seeds, like the stage-2 KI sweep:
# the baseline gets its settings chosen on the same seeds as the agent's checkpoints, never on test seeds.
STAGE3_ENV = {"pad_speed_range": (4.0, 8.33), "speed_limit": 10.0, "drag": True, "start_following": True,
              "pad_per_physics_step": True, "pad_accel_range": (1.0, 2.0), "own_velocity_obs": True,
              "action_rate_limit": 0.08, "pad_accel_obs": True}
GRID = {
    "kp": [2.4, 3.6, 4.8],  # first sweep: 1.2 < 2.4 (40% vs 56% averaged over the grid)
    "ki": [0.5, 1.0],  # needed again at speed (drag + the inner loop's velocity lag); first sweep: 0 -> 7%, 2 overshoots
    "accel_lead": [0.0, 0.3],  # s; 0 = ignore the broadcast acceleration; 0.6 was worse in the first sweep
    "align_radius": [0.15, 0.3],  # start descending when this close horizontally
    "steady_descent": [False, True],  # only descend while the vehicle is not changing speed
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep PID gains for stage 3 on validation seeds.")
    parser.add_argument("--seed", type=int, default=500, help="validation seeds; never tune on the test seeds")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--out", type=Path, default=Path("results/pid/val/stage3_tune.csv"))
    return parser.parse_args()


def evaluate_config(job: tuple[dict, int, int]) -> list[dict]:
    params, seed, episodes = job
    env = LandingAviary(**STAGE3_ENV)
    policy = PIDLander(speed_limit=env.SPEED_LIMIT, dt=env.CTRL_TIMESTEP, feedforward=True, **params)
    rows = [{**params, **run_episode(env, policy, seed + i, gui=False)} for i in range(episodes)]
    env.close()
    return rows


def main() -> None:
    args = parse_args()
    configs = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    with Pool(args.workers) as pool:
        results = pool.map(evaluate_config, [(params, args.seed, args.episodes) for params in configs])
    df = pd.DataFrame([row for rows in results for row in rows])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Same rule as the stage-2 PID: highest success, then the fastest
    summary = df.groupby(list(GRID)).agg(success=("success", "mean"), time_s=("duration_s", "median"),
                                         touchdown_max=("touchdown_speed", "max"), offset_max=("touchdown_offset", "max"),
                                         tilt_max=("max_tilt_deg", "max"))
    summary = summary.sort_values(["success", "time_s"], ascending=[False, True])
    print(summary.head(12).round(3).to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
