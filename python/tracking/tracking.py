import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass


@dataclass
class AxisConfig:
    max_rate_deg_s: float      # max angular speed at full stick deflection (deg/s)
    min_deg: float             # lower limit (deg)
    max_deg: float             # upper limit (deg)
    deadzone: float = 0.02     # ignore input below this (normalized)
    invert: bool = False       # flip the axis direction


# Intended mechanical limits. Intersected with the MuJoCo actuator ranges in init().
PAN_CONFIG = AxisConfig(max_rate_deg_s=60.0, min_deg=-90.0, max_deg=90.0)
TILT_CONFIG = AxisConfig(max_rate_deg_s=45.0, min_deg=-90.0, max_deg=90.0)

TRACK_MAX_RATE_DEG_S = 120.0   # slew limit for the setpoint while tracking
TRACK_LOST_GRACE_S = 0.35      # sim seconds


class Tracker:
    LEAD_S = 0.10              # ~ camera latency + actuator lag; NOT 0.7
    MAX_EXTRAP_S = 0.30
    SIGMA_ACCEL = 150.0        # deg/s^2 process noise: higher = more responsive, noisier
    MEAS_STD_DEG = 0.5         # measurement noise (deg)

    def __init__(self, model, data, fovy):
        self.looking = False
        self.pan_target = self.tilt_target = 0.0
        self.fovy = fovy
        self.data = data
        self.last_detection_time = 0.0

        self.last_meas = None                # latest latency-compensated (pan, tilt) in deg

        self.pose_hist = deque(maxlen=600)   # (t, pan_deg, tilt_deg)
        self.kf_ts = None                    # timestamp of last measurement fed to the KF
        self.kf_ready = False

        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], dtype=np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.MEAS_STD_DEG ** 2
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 10
        self.kf = kf

    # ---------------- public ----------------
    def init(self, model, data, pan_id, tilt_id):
        pan_lo, pan_hi = np.rad2deg(model.actuator_ctrlrange[pan_id])
        tilt_lo, tilt_hi = np.rad2deg(model.actuator_ctrlrange[tilt_id])
        PAN_CONFIG.min_deg = max(PAN_CONFIG.min_deg, float(pan_lo))
        PAN_CONFIG.max_deg = min(PAN_CONFIG.max_deg, float(pan_hi))
        TILT_CONFIG.min_deg = max(TILT_CONFIG.min_deg, float(tilt_lo))
        TILT_CONFIG.max_deg = min(TILT_CONFIG.max_deg, float(tilt_hi))
        self.data = data
        self.pan_target = self._pan_now()
        self.tilt_target = self._tilt_now()

    def update(self, dt_s, time_s, obs, cmd):
        self.pose_hist.append((time_s, self._pan_now(), self._tilt_now()))

        if obs is not None:
            new_looking = bool(obs["looking"])
            if new_looking != self.looking:
                print("TARGET TRACKING" if new_looking else "MANUAL AIMING")
            self.looking = new_looking
            if obs["detected"]:
                self.last_detection_time = obs["timestamp"]

        fresh = (obs is not None and obs["detected"] and obs["center"] is not None
                 and time_s - self.last_detection_time <= TRACK_LOST_GRACE_S)

        if self.looking and fresh:
            self._ingest(obs)                     # always keep the filter fed
            goal_pan, goal_tilt = self._estimate(time_s)
            self.pan_target = self._slew(self.pan_target, goal_pan,
                                         TRACK_MAX_RATE_DEG_S, dt_s, PAN_CONFIG)
            self.tilt_target = self._slew(self.tilt_target, goal_tilt,
                                          TRACK_MAX_RATE_DEG_S, dt_s, TILT_CONFIG)
        else:
            self.kf_ready = False                 # drop stale track
            self.pan_target = self._step_axis(self.pan_target, -cmd.x, PAN_CONFIG, dt_s)
            self.tilt_target = self._step_axis(self.tilt_target, cmd.y, TILT_CONFIG, dt_s)

    def pan(self) -> float:
        return self.pan_target

    def tilt(self) -> float:
        return self.tilt_target

    # ---------------- private ----------------
    def _pan_now(self) -> float:
        return float(np.rad2deg(self.data.joint("pan").qpos[0]))

    def _tilt_now(self) -> float:
        return float(np.rad2deg(self.data.joint("tilt").qpos[0]))

    def _pose_at(self, t):
        """Pose when the image was captured (latency compensation)."""
        best = None
        for ts, p, q in self.pose_hist:
            if ts <= t:
                best = (p, q)
            else:
                break
        return best if best is not None else (self._pan_now(), self._tilt_now())

    def _set_dt(self, dt):
        kf = self.kf
        kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                        [0, 1, 0, dt],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], dtype=np.float32)
        # white-noise-acceleration process model
        q = np.array([[dt ** 4 / 4, dt ** 3 / 2],
                      [dt ** 3 / 2, dt ** 2]], dtype=np.float32) * self.SIGMA_ACCEL ** 2
        Q = np.zeros((4, 4), dtype=np.float32)
        Q[np.ix_([0, 2], [0, 2])] = q
        Q[np.ix_([1, 3], [1, 3])] = q
        kf.processNoiseCov = Q

    def _ingest(self, obs):
        ts = obs["timestamp"]
        if ts == self.kf_ts:          # same frame as last time -> do NOT re-feed the filter
            return
        cx, cy = obs["center"]
        w, h = obs["width"], obs["height"]
        fovy = np.deg2rad(self.fovy)
        fy = (h / 2.0) / np.tan(fovy / 2)
        fx = fy                        # square pixels

        pan_obs, tilt_obs = self._pose_at(ts)
        meas_pan = pan_obs - np.rad2deg(np.arctan((cx - w / 2.0) / fx))
        meas_tilt = tilt_obs - np.rad2deg(np.arctan((cy - h / 2.0) / fy))
        self.last_meas = (float(meas_pan), float(meas_tilt))

        if not self.kf_ready or self.kf_ts is None or (ts - self.kf_ts) > TRACK_LOST_GRACE_S:
            self.kf.statePost = np.array([[meas_pan], [meas_tilt], [0], [0]], dtype=np.float32)
            self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10
            self.kf_ready = True
        else:
            self._set_dt(float(np.clip(ts - self.kf_ts, 1e-3, 0.2)))
            self.kf.predict()
            self.kf.correct(np.array([[meas_pan], [meas_tilt]], dtype=np.float32))
        self.kf_ts = ts

    def _estimate(self, time_s):
        s = self.kf.statePost.flatten()
        age = float(np.clip(time_s - self.kf_ts, 0.0, self.MAX_EXTRAP_S))
        horizon = age + self.LEAD_S
        return s[0] + s[2] * horizon, s[1] + s[3] * horizon

    @staticmethod
    def _slew(cur, goal, max_rate, dt, cfg):
        step = max_rate * dt
        cur = cur + float(np.clip(goal - cur, -step, step))
        return float(np.clip(cur, cfg.min_deg, cfg.max_deg))

    def _step_axis(self, target_deg, value, cfg, dt):
        """Integrate a normalized input [-1, 1] into a target angle (deg), clamped to limits."""
        value = float(np.clip(value, -1.0, 1.0))
        if abs(value) < cfg.deadzone:
            value = 0.0
        if cfg.invert:
            value = -value
        target_deg += value * cfg.max_rate_deg_s * dt
        return float(np.clip(target_deg, cfg.min_deg, cfg.max_deg))