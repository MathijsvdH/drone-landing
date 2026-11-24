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
# Following start (fast pad): the drone starts with the pad's velocity plus this error per horizontal axis (m/s),
# like the GPS-following phase before a real landing on a moving vehicle
FOLLOW_VEL_ERROR = 0.5

# Changing pad speed: the pad drives toward a new target speed (from pad_speed_range) every few seconds,
# speeding up or braking at an acceleration drawn per episode from pad_accel_range
SPEED_SEGMENT_SEC = (1.0, 3.0)

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
LANDING_REWARD = 100.0  # soft, centred touchdown (at the exact centre if the reward is graded, run 7)
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
                 action_rate_limit: float | None = None,
                 landing_reward_edge: float = LANDING_REWARD,
                 landing_speed_reward_edge: float = LANDING_REWARD,
                 bad_landing_penalty: float = BAD_LANDING_PENALTY,
                 drag: bool = False,
                 start_following: bool = False,
                 pad_per_physics_step: bool = False,
                 pad_accel_range: tuple[float, float] | None = None,
                 pad_accel_obs: bool = False,
                 relative_action_scale: float | None = None,
                 ):
        self.EPISODE_LEN_SEC = episode_len_sec
        self.pad_speed_range = pad_speed_range  # m/s, sampled per episode; (0, 0) = static pad
        # Fast-pad settings, all off by default (then episodes are exactly as before):
        # - start_following: the drone starts flying along with the pad instead of hovering
        # - pad_per_physics_step: move the pad every physics step (240 Hz) instead of jumping once per control step;
        #   at 8.3 m/s a control-step jump is 28 cm, which would make the pad leap away under the drone
        # - pad_accel_range: the pad speeds up and brakes (m/s^2) instead of driving at one constant speed
        self.start_following = start_following
        self.pad_per_physics_step = pad_per_physics_step
        self.pad_accel_range = pad_accel_range
        # Set before super().__init__, which already calls _observationSpace
        self.own_velocity_obs = own_velocity_obs
        # Max change of the velocity command per control step (m/s); None = commands applied as they come
        self.action_rate_limit = action_rate_limit
        self.applied_cmd = np.zeros(3)
        # Actions relative to the pad (stage 3, residual control): the action is a correction of at most this many m/s
        # on top of the pad's velocity, instead of an absolute velocity up to the speed limit. With speed limit 10 an
        # absolute action turns exploration noise into +-3 m/s; relative to the pad it stays at stage 2's scale.
        # None = absolute velocity actions, as before.
        self.relative_action_scale = relative_action_scale
        # The vehicle broadcasts its current acceleration (stage 3, like a datalink from a cooperative vehicle)
        self.pad_accel_obs = pad_accel_obs
        self.pad_acc = np.zeros(3)
        self.obs_size = 12 + 3 * own_velocity_obs + 3 * (action_rate_limit is not None) + 3 * pad_accel_obs
        self.pad_pos = np.array([0.0, 0.0, PAD_HEIGHT])  # centre of the pad's top surface
        self.pad_vel = np.zeros(3)
        # A landing only counts as "landed" under these limits. Training can make them stricter than the
        # milestone (a safety margin), so the agent's normal spread still passes the milestone at test time.
        self.touchdown_speed_limit = touchdown_speed_limit
        self.touchdown_offset_limit = touchdown_offset_limit
        # Precision ("archery board"): the landing reward drops linearly from LANDING_REWARD at the pad centre to
        # landing_reward_edge at the offset limit. Equal to LANDING_REWARD = the old flat reward.
        self.landing_reward_edge = landing_reward_edge
        # Softness (run 8): the same kind of cone on touchdown speed, LANDING_REWARD at 0 m/s down by
        # (LANDING_REWARD - landing_speed_reward_edge) at the speed limit. Both cones add up; the worst landing that
        # still counts (both at their limit) gets edge + speed edge - LANDING_REWARD, which must stay above a miss.
        self.landing_speed_reward_edge = landing_speed_reward_edge
        self.bad_landing_penalty = bad_landing_penalty
        self.world_state: int | None = None  # saved PyBullet world for the fast reset, set below
        if drag:
            physics = Physics.PYB_DRAG  # the library's drag model: air resistance grows with speed and rotor speed
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
        self.pad_heading = np.array([np.cos(heading), np.sin(heading), 0.0])
        self.pad_time = 0.0
        self.pad_acc = np.zeros(3)
        # Extra draws only when a setting is on, after all the others, so default episodes don't change
        follow_error = np.zeros(3)
        if self.start_following:
            follow_error[0:2] = self.np_random.uniform(-FOLLOW_VEL_ERROR, FOLLOW_VEL_ERROR, size=2)
            # Start the pad half an episode's drive before the origin, so a fast pad stays on the ground plane
            # (its collision box is 200 x 200 m; 8.3 m/s x 15 s = 125 m)
            self.pad_pos = self.pad_pos - self.pad_vel * self.EPISODE_LEN_SEC / 2
        if self.pad_accel_range is not None:
            self.pad_accel = self.np_random.uniform(*self.pad_accel_range)
            self.speed_schedule = self._drawSpeedSchedule()
        self.INIT_XYZS = np.array([[self.pad_pos[0] + xy[0], self.pad_pos[1] + xy[1], z]])
        if self.world_state is None:
            super()._housekeeping()  # library: zero its counters, (re)load plane, drone and pad
        else:
            self._housekeepingWithoutReload()
        start_vel = self.pad_vel + follow_error if self.start_following else np.zeros(3)
        if self.start_following:
            p.resetBaseVelocity(self.DRONE_IDS[0], linearVelocity=start_vel.tolist(), physicsClientId=self.CLIENT)
            self._movePadBody()
        self.ctrl[0].reset()  # the library never resets the PID, so integral errors leak between episodes
        self.prev_dist = float(np.linalg.norm(self.pad_pos - self.INIT_XYZS[0]))
        self.prev_rel_vel = np.zeros(3)
        self.outcome = "flying"
        self.reward_terms = self._zeroRewardTerms()
        self.touchdown_speed: float | None = None  # set on first contact with the pad
        self.touchdown_offset: float | None = None
        self.too_fast = False  # why a touchdown was a bad landing (both can be true)
        self.off_centre = False
        # The rate limiter starts from the drone's start velocity (0 from hover, ~ the pad's velocity when following)
        self.applied_cmd = start_vel.copy()

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
        if not self.pad_per_physics_step:
            self._advancePad(self.CTRL_TIMESTEP)
        if self.relative_action_scale is not None:
            action = self._relativeToAbsolute(action)
        if self.action_rate_limit is not None:
            action = self._limitActionRate(action)
        return super().step(action)

    def _physics(self, rpm, nth_drone):
        # The library calls this once per physics step, right before stepSimulation (also with drag on)
        if self.pad_per_physics_step:
            self._advancePad(self.PYB_TIMESTEP)
        super()._physics(rpm, nth_drone)

    def _advancePad(self, dt: float) -> None:
        if self.pad_accel_range is not None:
            self._updatePadSpeed(dt)
        self.pad_pos = self.pad_pos + self.pad_vel * dt
        self._movePadBody()

    def _movePadBody(self) -> None:
        p.resetBasePositionAndOrientation(self.PAD_ID, [self.pad_pos[0], self.pad_pos[1], PAD_HEIGHT / 2],
                                          [0, 0, 0, 1], physicsClientId=self.CLIENT)

    def _drawSpeedSchedule(self) -> list[tuple[float, float]]:
        # (start time, target speed) per segment, drawn at reset so a seed fixes the whole speed profile
        schedule, t = [], 0.0
        while t < self.EPISODE_LEN_SEC:
            t += self.np_random.uniform(*SPEED_SEGMENT_SEC)
            schedule.append((t, self.np_random.uniform(*self.pad_speed_range)))
        return schedule

    def _updatePadSpeed(self, dt: float) -> None:
        # Drive toward the current segment's target speed at the episode's acceleration; heading stays the same
        self.pad_time += dt
        target = next((v for start, v in reversed(self.speed_schedule) if start <= self.pad_time),
                      float(np.linalg.norm(self.pad_vel)))
        speed = float(np.linalg.norm(self.pad_vel))
        change = float(np.clip(target - speed, -self.pad_accel * dt, self.pad_accel * dt))
        self.pad_vel = (speed + change) * self.pad_heading
        self.pad_acc = change / dt * self.pad_heading  # what the vehicle broadcasts: +-pad_accel, or 0 at its target

    def _relativeToAbsolute(self, action: np.ndarray) -> np.ndarray:
        # Same action format as the library's VEL action, read as a correction on top of the pad's velocity (what a
        # flight stack's guidance layer does with the target's velocity from the datalink). Output: the library's
        # format for the absolute velocity, so the rate limiter and the inner loop work exactly as before.
        a = np.asarray(action, dtype=float).reshape(4)
        norm = np.linalg.norm(a[0:3])
        correction = self.relative_action_scale * abs(a[3]) * (a[0:3] / norm if norm > 0 else np.zeros(3))
        velocity = self.pad_vel + correction
        speed = np.linalg.norm(velocity)
        if speed == 0:
            return np.zeros((1, 4))
        return np.hstack([velocity / speed, min(speed / self.SPEED_LIMIT, 1.0)]).reshape(1, 4)

    def _limitActionRate(self, action: np.ndarray) -> np.ndarray:
        # Safety layer between policy and drone: the velocity command may change by at most action_rate_limit per
        # step (0.08 m/s per 33 ms = 2.4 m/s^2). Faster changes wind up the library's lagging inner loop (ramp test,
        # 2026-11-20). The library reads a VEL action as velocity = SPEED_LIMIT * |a[3]| * unit(a[0:3]).
        a = np.asarray(action, dtype=float).reshape(4)
        norm = np.linalg.norm(a[0:3])
        target = self.SPEED_LIMIT * abs(a[3]) * (a[0:3] / norm if norm > 0 else np.zeros(3))
        change = target - self.applied_cmd
        size = np.linalg.norm(change)
        if size > self.action_rate_limit:
            change *= self.action_rate_limit / size
        self.applied_cmd = self.applied_cmd + change
        # Back to the library's action format
        speed = np.linalg.norm(self.applied_cmd)
        if speed == 0:
            return np.zeros((1, 4))
        return np.hstack([self.applied_cmd / speed, speed / self.SPEED_LIMIT]).reshape(1, 4)

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
        if self.action_rate_limit is not None:
            # With the limiter, what the drone does depends on the command applied last step, so the policy has to
            # see it (Markov property). Appended last, so the earlier values keep their meaning.
            # With relative actions it is shown in the same frame as the action (relative to the pad).
            relative = self.relative_action_scale is not None
            parts.append(self.applied_cmd - self.pad_vel if relative else self.applied_cmd)
        if self.pad_accel_obs:
            # The pad's acceleration can't be inferred from one snapshot; without it a braking vehicle is only noticed
            # once the relative velocity has already built up. Appended last, so the earlier values keep their meaning.
            parts.append(self.pad_acc)
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
            terms["landing"] = self._landingReward(self.touchdown_offset, self.touchdown_speed)
        elif self.outcome == "bad_landing":
            terms["bad_landing"] = self.bad_landing_penalty
        elif self.outcome in ("crashed", "out_of_bounds"):
            terms["crash"] = CRASH_PENALTY

        self.prev_dist = dist
        self.prev_rel_vel = rel_vel
        self.reward_terms = terms
        return float(sum(terms.values()))

    def _landingReward(self, offset: float, speed: float) -> float:
        reward = LANDING_REWARD  # flat reward when both edges are left at LANDING_REWARD, exactly as before
        if self.landing_reward_edge != LANDING_REWARD:
            # Linear cone, not rings: every centimetre closer pays, also near the centre
            fraction = offset / self.touchdown_offset_limit  # 0 at the centre, < 1 for every landing that counts
            reward -= (LANDING_REWARD - self.landing_reward_edge) * fraction
        if self.landing_speed_reward_edge != LANDING_REWARD:
            # Linear from 0 m/s, not a bonus below some threshold: every bit softer pays, also far under the limit
            reward -= (LANDING_REWARD - self.landing_speed_reward_edge) * speed / self.touchdown_speed_limit
        return reward

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
