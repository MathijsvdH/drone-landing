import time

import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics

# Pad: flat box lying on the ground plane (metres)
PAD_SIZE = 0.5
PAD_HEIGHT = 0.05

# Start position relative to the pad centre
START_XY_RANGE = 1.0  # horizontal offset sampled uniformly in [-range, +range]
START_Z_RANGE = (1.0, 1.5)  # height above the ground

# Success criteria (milestone 1). Evaluation always uses these; training can use stricter ones
MAX_TOUCHDOWN_SPEED = 0.5  # m/s, relative to the pad
MAX_TOUCHDOWN_OFFSET = 0.10  # m, horizontal distance to the pad centre

# Crash / out-of-bounds limits
MAX_TILT = 1.0  # rad (~57 deg) on roll or pitch; the built-in PID recovers fine from ~35 deg at 1 m/s
MAX_HORIZONTAL_DIST = 3.0  # m from the pad centre
MAX_HEIGHT = 3.0  # m above the pad

# Reward terms
PROGRESS_WEIGHT = 10.0  # x (previous - current distance); potential-based, so the optimal policy is unchanged
TIME_PENALTY = -0.01  # every step, so hovering forever is not free
LANDING_REWARD = 100.0  # soft, centred touchdown
BAD_LANDING_PENALTY = -20.0  # touched the pad too fast or off-centre; mild so the agent doesn't avoid the pad
CRASH_PENALTY = -100.0  # ground contact, too much tilt, or out of bounds


class LandingAviary(BaseRLAviary):

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui: bool = False,
                 record: bool = False,
                 speed_limit: float = 1.0,
                 episode_len_sec: float = 15.0,
                 pad_speed_range: tuple[float, float] = (0.0, 0.0),
                 own_velocity_obs: bool = False,
                 touchdown_speed_limit: float = MAX_TOUCHDOWN_SPEED,
                 touchdown_offset_limit: float = MAX_TOUCHDOWN_OFFSET,
                 fast_reset: bool = True,
                 ):
        self.EPISODE_LEN_SEC = episode_len_sec
        self.pad_speed_range = pad_speed_range  # m/s, sampled per episode; (0, 0) = static pad
        # Set before super().__init__, which already calls _observationSpace
        self.own_velocity_obs = own_velocity_obs
        self.obs_size = 15 if own_velocity_obs else 12
        self.pad_pos = np.array([0.0, 0.0, PAD_HEIGHT])  # centre of the pad's top surface
        self.pad_vel = np.zeros(3)
        # A landing only counts as "landed" under these limits. Training can make them stricter than the
        # milestone (a safety margin), so the agent's normal spread still passes the milestone at test time.
        self.touchdown_speed_limit = touchdown_speed_limit
        self.touchdown_offset_limit = touchdown_offset_limit
        self.world_state: int | None = None  # saved PyBullet world for the fast reset, set below
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=ObservationType.KIN,
                         act=ActionType.VEL,
                         )
        self.SPEED_LIMIT = speed_limit  # library default is ~0.25 m/s, too slow to follow a moving pad
        # Not with GUI/recording: the library sets up its debug items and video logging in the full reset
        if fast_reset and not gui and not record:
            self.world_state = p.saveState(physicsClientId=self.CLIENT)

    def reset(self, seed=None, options=None):
        # The library's reset deletes the PyBullet world and reloads the plane and drone URDFs (~18 ms, as long
        # as ~30 control steps, so a third of the env time once episodes are ~2 s). The fast reset restores the
        # world saved right after it was built instead (same bodies, no contacts, zero velocities) and only moves
        # the drone to its new start. Checked: identical episodes, eval CSVs and training checkpoints.
        if self.world_state is None:
            return super().reset(seed=seed, options=options)
        gym.Env.reset(self, seed=seed, options=options)  # what the library's reset does first: seeds np_random
        p.restoreState(self.world_state, physicsClientId=self.CLIENT)
        self._housekeeping()
        self._updateAndStoreKinematicInformation()
        return self._computeObs(), self._computeInfo()

    def _housekeeping(self):
        # Called by the library on reset, after np_random is seeded
        xy = self.np_random.uniform(-START_XY_RANGE, START_XY_RANGE, size=2)
        z = self.np_random.uniform(*START_Z_RANGE)
        # Drawn after the start position, so a static pad gives exactly the same episodes as before
        heading = self.np_random.uniform(0, 2 * np.pi)
        speed = self.np_random.uniform(*self.pad_speed_range)
        self.pad_pos = np.array([0.0, 0.0, PAD_HEIGHT])
        self.pad_vel = speed * np.array([np.cos(heading), np.sin(heading), 0.0])
        self.INIT_XYZS = np.array([[self.pad_pos[0] + xy[0], self.pad_pos[1] + xy[1], z]])
        if self.world_state is None:
            super()._housekeeping()  # library: zero its counters, (re)load plane, drone and pad
        else:
            self._housekeepingWithoutReload()
        self.ctrl[0].reset()  # the library never resets the PID, so integral errors leak between episodes
        self.prev_dist = float(np.linalg.norm(self.pad_pos - self.INIT_XYZS[0]))
        self.prev_rel_vel = np.zeros(3)
        self.outcome = "flying"
        self.reward_terms = self._zeroRewardTerms()
        self.touchdown_speed: float | None = None  # set on first contact with the pad
        self.touchdown_offset: float | None = None
        self.too_fast = False  # why a touchdown was a bad landing (both can be true)
        self.off_centre = False

    def _housekeepingWithoutReload(self):
        # The parts of BaseAviary._housekeeping that step() relies on, without reloading the world.
        # GUI-only fields stay as set by the first (full) housekeeping, since the fast reset is off with GUI.
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        # restoreState already put the pad back at the origin and the drone at rest; move the drone to its start
        p.resetBasePositionAndOrientation(self.DRONE_IDS[0], self.INIT_XYZS[0],
                                          p.getQuaternionFromEuler(self.INIT_RPYS[0]), physicsClientId=self.CLIENT)

    def _addObstacles(self):
        half = [PAD_SIZE / 2, PAD_SIZE / 2, PAD_HEIGHT / 2]
        self.PAD_ID = p.createMultiBody(
            baseMass=0,  # mass 0 = static body in PyBullet
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=half,
                                                           physicsClientId=self.CLIENT),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                                     rgbaColor=[0.2, 0.6, 0.2, 1],
                                                     physicsClientId=self.CLIENT),
            basePosition=[self.pad_pos[0], self.pad_pos[1], PAD_HEIGHT / 2],
            physicsClientId=self.CLIENT,
        )

    def step(self, action):
        # Move the pad before the physics steps, so the observation and reward at the end of this
        # step see the pad where it was during the step. The pad is teleported (it stays a static
        # body), which is fine because the episode ends at the first contact anyway.
        self.pad_pos = self.pad_pos + self.pad_vel * self.CTRL_TIMESTEP
        p.resetBasePositionAndOrientation(self.PAD_ID, [self.pad_pos[0], self.pad_pos[1], PAD_HEIGHT / 2],
                                          [0, 0, 0, 1], physicsClientId=self.CLIENT)
        return super().step(action)

    def _observationSpace(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.obs_size), dtype=np.float32)

    def _computeObs(self) -> np.ndarray:
        state = self._getDroneStateVector(0)
        rel_pos = self.pad_pos - state[0:3]
        rel_vel = self.pad_vel - state[10:13]
        rpy, ang_vel = state[7:10], state[13:16]
        parts = [rel_pos, rel_vel, rpy, ang_vel]
        if self.own_velocity_obs:
            # The drone's own velocity (ego-motion, what an IMU / optical flow gives), not a world position.
            # With it the policy can work out the pad's velocity: pad_vel = rel_vel + own_vel.
            # Appended last, so values 0-11 mean the same as in the 12-value observation.
            parts.append(state[10:13])
        return np.hstack(parts).reshape(1, self.obs_size).astype(np.float32)

    def _computeReward(self) -> float:
        # The library calls this first after each step, so the outcome is decided here
        # and reused by _computeTerminated, _computeTruncated and _computeInfo
        state = self._getDroneStateVector(0)
        rel_pos = self.pad_pos - state[0:3]
        rel_vel = self.pad_vel - state[10:13]
        dist = float(np.linalg.norm(rel_pos))

        self.outcome = self._classifyOutcome(state, rel_pos)
        terms = self._zeroRewardTerms()
        terms["progress"] = PROGRESS_WEIGHT * (self.prev_dist - dist)
        terms["time"] = TIME_PENALTY
        if self.outcome == "landed":
            terms["landing"] = LANDING_REWARD
        elif self.outcome == "bad_landing":
            terms["bad_landing"] = BAD_LANDING_PENALTY
        elif self.outcome in ("crashed", "out_of_bounds"):
            terms["crash"] = CRASH_PENALTY

        self.prev_dist = dist
        self.prev_rel_vel = rel_vel
        self.reward_terms = terms
        return float(sum(terms.values()))

    def _classifyOutcome(self, state: np.ndarray, rel_pos: np.ndarray) -> str:
        if p.getContactPoints(bodyA=int(self.DRONE_IDS[0]), bodyB=self.PAD_ID,
                              physicsClientId=self.CLIENT):
            # Contact resolution has already slowed the drone down by the end of this step,
            # so use the relative velocity from the previous control step as impact speed
            self.touchdown_speed = float(np.linalg.norm(self.prev_rel_vel))
            self.touchdown_offset = float(np.linalg.norm(rel_pos[0:2]))
            self.too_fast = self.touchdown_speed >= self.touchdown_speed_limit
            self.off_centre = self.touchdown_offset >= self.touchdown_offset_limit
            return "bad_landing" if self.too_fast or self.off_centre else "landed"
        if p.getContactPoints(bodyA=int(self.DRONE_IDS[0]), bodyB=self.PLANE_ID,
                              physicsClientId=self.CLIENT):
            return "crashed"
        if abs(state[7]) > MAX_TILT or abs(state[8]) > MAX_TILT:
            return "crashed"
        if np.linalg.norm(rel_pos[0:2]) > MAX_HORIZONTAL_DIST or -rel_pos[2] > MAX_HEIGHT:
            return "out_of_bounds"
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return "timeout"
        return "flying"

    def _computeTerminated(self) -> bool:
        return self.outcome in ("landed", "bad_landing", "crashed", "out_of_bounds")

    def _computeTruncated(self) -> bool:
        return self.outcome == "timeout"

    def _computeInfo(self) -> dict:
        return {
            "outcome": self.outcome,
            "touchdown_speed": self.touchdown_speed,
            "touchdown_offset": self.touchdown_offset,
            "too_fast": self.too_fast,
            "off_centre": self.off_centre,
            "distance": self.prev_dist,
            **{f"reward/{name}": value for name, value in self.reward_terms.items()},
        }

    @staticmethod
    def _zeroRewardTerms() -> dict[str, float]:
        return {"progress": 0.0, "time": 0.0, "landing": 0.0, "bad_landing": 0.0, "crash": 0.0}
