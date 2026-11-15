import numpy as np

# The baseline only sees the same observation as the agent it is compared with (12 values, or 15 with
# feedforward for run 4) and outputs the same VEL action, so the comparison is about the decision making,
# not about having more information.
# Like the agent, it gives a velocity command; the library's DSLPIDControl turns that into motor RPMs.

# Horizontal PID on the position error to the pad (output: desired horizontal velocity, m/s)
KP_XY = 1.2
KI_XY = 1.0  # needed for a moving pad: P alone must trail the pad to command its speed (steady-state error)
KI_XY_FEEDFORWARD = 0.0  # with feedforward the pad velocity is known, so I only adds overshoot (seeds 500-599)
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

    # The defaults are the tuned baseline; the other arguments exist for the speed-success sweep (pid_sweep.py)
    # feedforward needs the 15-value observation (own velocity), the same information the run-4 agent gets
    def __init__(self, speed_limit: float, dt: float, kp: float = KP_XY, ki: float | None = None,
                 align_radius: float = ALIGN_RADIUS, descent_gain: float = DESCENT_GAIN,
                 min_descent: float = MIN_DESCENT, max_descent: float = MAX_DESCENT, feedforward: bool = False):
        self.speed_limit = speed_limit
        self.dt = dt
        self.feedforward = feedforward
        if ki is None:
            ki = KI_XY_FEEDFORWARD if feedforward else KI_XY
        self.xy = PID(kp, ki, KD_XY, I_LIMIT_XY)
        self.align_radius = align_radius
        self.descent_gain = descent_gain
        self.min_descent = min_descent
        self.max_descent = max_descent

    def reset(self) -> None:
        self.xy.reset()

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.reshape(-1)
        rel_pos, rel_vel = obs[0:3], obs[3:6]  # pad minus drone, so rel_pos is the error to drive to zero
        # The derivative of the position error is exactly the relative velocity, so the D term
        # can use it directly instead of differencing noisy positions
        v_xy = self.xy.update(rel_pos[0:2], rel_vel[0:2], self.dt)
        if self.feedforward:
            if obs.size != 15:
                raise ValueError("feedforward needs the 15-value observation (own_velocity_obs=True)")
            # rel_vel = pad_vel - own_vel, so the pad's velocity follows directly. Commanding it makes the
            # drone move with the pad; the PID then only has to correct the remaining position error.
            pad_vel = rel_vel + obs[12:15]
            v_xy = v_xy + pad_vel[0:2]

        height = -rel_pos[2]
        alignment = max(0.0, 1.0 - np.linalg.norm(rel_pos[0:2]) / self.align_radius)
        descent = np.clip(self.descent_gain * height, self.min_descent, self.max_descent)
        v_z = -alignment * descent

        return self._toAction(np.array([v_xy[0], v_xy[1], v_z]))

    def _toAction(self, velocity: np.ndarray) -> np.ndarray:
        # VEL action = [direction (3), magnitude as fraction of the speed limit], all in [-1, 1]
        speed = float(np.linalg.norm(velocity))
        if speed == 0.0:
            return np.zeros((1, 4), dtype=np.float32)
        magnitude = min(speed / self.speed_limit, 1.0)
        return np.hstack([velocity / speed, magnitude]).reshape(1, 4).astype(np.float32)
