import argparse
import itertools
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from envs.landing_aviary import LandingAviary
from evaluate import run_episode
from pid_baseline import PIDLander

# Speed-success trade-off of the PID baseline: from the tuned (patient) settings to aggressive ones.
# Answers "is the PID still better than the agent when it has to land as fast?". Validation seeds only.
GRID = {
    "kp": [1.2, 2.4],  # close the horizontal gap faster
    "align_radius": [0.15, 0.3, 0.6],  # start descending earlier, while still catching up
    "descent_gain": [0.6, 1.2, 2.4],  # descend faster when high
    "max_descent": [0.5, 1.0],
    "min_descent": [0.15, 0.3],  # final approach speed, roughly the touchdown speed
}
PAD_SPEED = (0.2, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep PID descent settings and write per-episode results.")
    parser.add_argument("--seed", type=int, default=500, help="validation seeds; never sweep on the test seeds")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--feedforward", action="store_true",
                        help="15-value observation + pad-velocity feedforward (the PID that matches run 4)")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: results/pid/tradeoff_val.csv (tradeoff_val_feedforward.csv with --feedforward)")
    args = parser.parse_args()
    args.out = args.out or Path(f"results/pid/tradeoff_val{'_feedforward' if args.feedforward else ''}.csv")
    return args


def evaluate_config(job: tuple[dict, int, int, bool]) -> list[dict]:
    params, seed, episodes, feedforward = job
    env = LandingAviary(gui=False, pad_speed_range=PAD_SPEED, own_velocity_obs=feedforward)
    policy = PIDLander(speed_limit=env.SPEED_LIMIT, dt=env.CTRL_TIMESTEP, feedforward=feedforward, **params)
    rows = [{**params, **run_episode(env, policy, seed + i, gui=False)} for i in range(episodes)]
    env.close()
    return rows


def main() -> None:
    args = parse_args()
    configs = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    jobs = [(params, args.seed, args.episodes, args.feedforward) for params in configs]
    with Pool(args.workers) as pool:
        results = pool.map(evaluate_config, jobs)

    df = pd.DataFrame([row for rows in results for row in rows])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    landed = df[df["success"]]
    summary = df.groupby(list(GRID)).agg(success=("success", "mean")).join(
        landed.groupby(list(GRID)).agg(time_s=("duration_s", "median"),
                                       touchdown_speed=("touchdown_speed", "median")))
    print(summary.sort_values("time_s").round(3).to_string())
    print(f"\n{len(configs)} configs x {args.episodes} episodes, wrote {args.out}")


if __name__ == "__main__":
    main()
