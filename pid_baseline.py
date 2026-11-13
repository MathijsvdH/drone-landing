import numpy as np

# The baseline only sees the same 12-value observation as the agent and outputs the same VEL action,
# so the comparison is about the decision making, not about having more information.
# Like the agent, it gives a velocity command; the library's DSLPIDControl turns that into motor RPMs.

# Horizontal PID on the position error to the pad (output: desired horizontal velocity, m/s)
KP_XY = 1.2
KI_XY = 1.0  # needed for a moving pad: P alone must trail the pad to command its speed (steady-state error)
KD_XY = 0.0
I_LIMIT_XY = 1.0  # anti-windup: cap the integral so it can't build up a huge command while far away

# Descent: only descend when horizontally lined up, slower close to the pad for a soft touchdown
ALIGN_RADIUS = 0.15  # m; no descent at all beyond this horizontal error, full descent rate at 0 error
DESCENT_GAIN = 0.6  # descent speed per metre of height above the pad
MIN_DESCENT = 0.15  # m/s; keep moving down at the end instead of hovering just above the pad
MAX_DESCENT = 0.5  # m/s; well under the 1.0 m/s speed limit


class PID:

    def __init__(self, kp: float, ki: float, kd: float, i_limit: float):
        self.kp, self.ki, self.kd, self.i_limit = kp, ki, kd, i_limit
        self.integral = np.zeros(2)

    def reset(self) -> None:
        self.integral = np.zeros(2)

    def update(self, error: np.ndarray, error_rate: np.ndarray, dt: float) -> np.ndarray:
        self.integral = np.clip(self.integral + error * dt, -self.i_limit, self.i_limit)
        return self.kp * error + self.ki * self.integral + self.kd * error_rate


class PIDLander:

    def __init__(self, speed_limit: float, dt: float):
        self.speed_limit = speed_limit
        self.dt = dt
        self.xy = PID(KP_XY, KI_XY, KD_XY, I_LIMIT_XY)

    def reset(self) -> None:
        self.xy.reset()

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.reshape(12)
        rel_pos, rel_vel = obs[0:3], obs[3:6]  # pad minus drone, so rel_pos is the error to drive to zero
        # The derivative of the position error is exactly the relative velocity, so the D term
        # can use it directly instead of differencing noisy positions
        v_xy = self.xy.update(rel_pos[0:2], rel_vel[0:2], self.dt)

        height = -rel_pos[2]
        alignment = max(0.0, 1.0 - np.linalg.norm(rel_pos[0:2]) / ALIGN_RADIUS)
        descent = np.clip(DESCENT_GAIN * height, MIN_DESCENT, MAX_DESCENT)
        v_z = -alignment * descent

        return self._toAction(np.array([v_xy[0], v_xy[1], v_z]))

    def _toAction(self, velocity: np.ndarray) -> np.ndarray:
        # VEL action = [direction (3), magnitude as fraction of the speed limit], all in [-1, 1]
        speed = float(np.linalg.norm(velocity))
        if speed == 0.0:
            return np.zeros((1, 4), dtype=np.float32)
        magnitude = min(speed / self.speed_limit, 1.0)
        return np.hstack([velocity / speed, magnitude]).reshape(1, 4).astype(np.float32)
