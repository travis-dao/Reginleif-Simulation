# mjpython python/tracking/main.py

import time
import queue
import multiprocessing as mp
from dataclasses import dataclass

import sys
from pathlib import Path

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import numpy as np
import mujoco
import mujoco.viewer

from input.input import InputController
from input.visualizer import InputVisualizer, StateVisualizer

from tracking.util import rand_pos_cube, lerp


@dataclass
class AxisConfig:
    max_rate_deg_s: float      # max angular speed at full stick deflection (deg/s)
    min_deg: float             # lower limit (deg)
    max_deg: float             # upper limit (deg)
    deadzone: float = 0.02     # ignore input below this (normalized)
    invert: bool = False       # flip the axis direction


# Intended mechanical limits. Intersected with the MuJoCo actuator ranges below.
# PAN_CONFIG = AxisConfig(max_rate_deg_s=60.0, min_deg=-30.0, max_deg=30.0)
# TILT_CONFIG = AxisConfig(max_rate_deg_s=45.0, min_deg=-15.0, max_deg=15.0)
PAN_CONFIG = AxisConfig(max_rate_deg_s=60.0, min_deg=-90.0, max_deg=90.0)
TILT_CONFIG = AxisConfig(max_rate_deg_s=45.0, min_deg=-90.0, max_deg=90.0)

CAMERA_FPS = 30

# Image-space tracking gains. Output is a desired angular rate, not a servo torque.
TRACK_MAX_RATE_DEG_S = 35.0
TRACK_DEADBAND = 0.025       # fraction of half-image width/height
TRACK_LOST_GRACE_S = 0.35

K_P = 10


def step_axis(target_deg: float, value: float, cfg: AxisConfig, dt: float) -> float:
    """Integrate a normalized input [-1, 1] into a target angle (deg), clamped to limits."""
    value = float(np.clip(value, -1.0, 1.0))
    if abs(value) < cfg.deadzone:
        value = 0.0
    if cfg.invert:
        value = -value
    target_deg += value * cfg.max_rate_deg_s * dt
    return float(np.clip(target_deg, cfg.min_deg, cfg.max_deg))


def _put_latest(q, item):
    """Keep inter-process messages fresh instead of building a stale backlog."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


def camera_window(frame_q, event_q):
    """Detect objects and publish the latest image-space target observation.

    A target is acquired when the user centers the crosshair on it for the
    configured dwell time. Once acquired, tracking stays active while the
    object remains detectable; the target does not need to stay centered.
    """
    import cv2
    import obj_detection as det

    det.create_config_window()
    cv2.namedWindow(det.VIEW_WINDOW, cv2.WINDOW_NORMAL)
    tracker = det.DwellTracker()
    show_mask = False

    while True:
        frame = frame_q.get()
        if frame is None:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        config = det.get_config()
        bbox, mask = det.find_object(img, config)
        crosshair = det.crosshair_point(img.shape)
        zone = det.object_zone(bbox) if bbox is not None else None
        hit = zone is not None and det.in_zone(crosshair, zone)
        now = time.monotonic()
        progress, started, stopped = tracker.update(
            hit=hit, detected=bbox is not None, now=now,
            dwell_s=config["dwell_s"],
        )

        h, w = img.shape[:2]
        observation = {
            "detected": bbox is not None,
            "center": ((bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0) if bbox is not None else None),
            "width": w,
            "height": h,
            "looking": tracker.looking,
            "progress": progress,
            "timestamp": now,
        }
        _put_latest(event_q, observation)

        if started:
            print("TARGET ACQUIRED — automatic tracking enabled")
        if stopped:
            print("TARGET LOST — returned to manual control")

        cv2.imshow(det.VIEW_WINDOW, det.draw_overlay(
            img, bbox, crosshair, zone, progress, tracker.looking
        ))
        if show_mask:
            cv2.imshow(det.MASK_WINDOW, mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow(det.MASK_WINDOW)
        if key == ord("u"):
            tracker.unlock()
        if key == ord("p"):
            print(f"HSV lower={config['lower'].tolist()} upper={config['upper'].tolist()}")

    cv2.destroyAllWindows()


def main():
    input_controller = InputController()
    # input_visualizer = InputVisualizer()
    state_visualizer = StateVisualizer()

    model = mujoco.MjModel.from_xml_path("models/cam/pan_tilt.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'tz_cam')
    pan_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pan_servo")
    tilt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tilt_servo")

    target_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "green_block"
    )
    mocap_id = model.body_mocapid[target_body_id]
    target_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "target"
    )

    if mocap_id < 0:
        raise RuntimeError("green_block is not a mocap body")

    if pan_id < 0 or tilt_id < 0:
        raise RuntimeError("Could not find pan_servo and tilt_servo actuators in the MuJoCo model")

    pan_lo, pan_hi = np.rad2deg(model.actuator_ctrlrange[pan_id])
    tilt_lo, tilt_hi = np.rad2deg(model.actuator_ctrlrange[tilt_id])
    PAN_CONFIG.min_deg = max(PAN_CONFIG.min_deg, float(pan_lo))
    PAN_CONFIG.max_deg = min(PAN_CONFIG.max_deg, float(pan_hi))
    TILT_CONFIG.min_deg = max(TILT_CONFIG.min_deg, float(tilt_lo))
    TILT_CONFIG.max_deg = min(TILT_CONFIG.max_deg, float(tilt_hi))

    pan_target = float(np.rad2deg(data.joint("pan").qpos[0]))
    tilt_target = float(np.rad2deg(data.joint("tilt").qpos[0]))

    frame_q = mp.Queue(maxsize=2)
    event_q = mp.Queue(maxsize=2)  # latest observation wins
    proc = mp.Process(target=camera_window, args=(frame_q, event_q), daemon=True)
    proc.start()

    dt = model.opt.timestep
    render_every = max(1, round(1 / CAMERA_FPS / dt))
    step = 0
    looking = False
    latest_observation = None
    last_detection_time = 0.0

    box_initial_pos = data.mocap_pos[mocap_id].copy()
    box_start_pos = box_initial_pos
    box_target_pos = box_initial_pos
    box_move_start_time = data.time
    box_move_duration = 1.0

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and proc.is_alive():
                t0 = time.monotonic()
                input_controller.update(dt)
                cmd = input_controller.get_input()
                # input_visualizer.update(cmd)
                error_x = error_y = pan_error_deg = tilt_error_deg = None

                # Consume all queued observations and retain only the newest.
                while True:
                    try:
                        latest_observation = event_q.get_nowait()
                    except queue.Empty:
                        break

                if latest_observation is not None:
                    new_looking = bool(latest_observation["looking"])
                    if new_looking != looking:
                        print("TARGET TRACKING" if new_looking else "MANUAL AIMING")
                    looking = new_looking
                    if latest_observation["detected"]:
                        last_detection_time = latest_observation["timestamp"]

                if looking and latest_observation is not None:
                    center = latest_observation["center"]
                    width = latest_observation["width"]
                    height = latest_observation["height"]
                    fresh = (latest_observation["detected"] and
                             time.monotonic() - last_detection_time <= TRACK_LOST_GRACE_S)
                    if fresh and center is not None:
                        # Normalize image error to [-1, 1]. Positive x means the
                        # target is right of center; positive y means it is below.
                        # ex = (center[0] - width / 2.0) / max(width / 2.0, 1.0)
                        # ey = (center[1] - height / 2.0) / max(height / 2.0, 1.0)
                        error_x = (center[0] - width / 2.0) 
                        error_y = (center[1] - height / 2.0) 

                        fovy = model.cam_fovy[cam_id]  # MuJoCo camera vertical FOV, in degrees
                        fovx = 2 * np.arctan(np.tan(np.deg2rad(fovy) / 2) * (width / height))

                        fx = (width / 2.0) / np.tan(fovx / 2)
                        fy = (height / 2.0) / np.tan(np.deg2rad(fovy) / 2)

                        pan_error_deg = -np.rad2deg(np.arctan(error_x / fx))
                        tilt_error_deg = -np.rad2deg(np.arctan(error_y / fy))

                        pan_error_deg = np.clip(pan_error_deg, -TRACK_MAX_RATE_DEG_S, TRACK_MAX_RATE_DEG_S)
                        tilt_error_deg = np.clip(tilt_error_deg, -TRACK_MAX_RATE_DEG_S, TRACK_MAX_RATE_DEG_S)
                        
                        pan_target = step_axis(pan_target + (pan_error_deg * K_P * dt), -cmd.x, PAN_CONFIG, dt)
                        tilt_target = step_axis(tilt_target + (tilt_error_deg * K_P * dt), cmd.y, TILT_CONFIG, dt)
                else:
                    # Manual input remains available before acquisition and
                    # immediately after target loss.
                    pan_target = step_axis(pan_target, -cmd.x, PAN_CONFIG, dt)
                    tilt_target = step_axis(tilt_target, cmd.y, TILT_CONFIG, dt)

                pan_angle_deg = np.rad2deg(data.joint("pan").qpos[0])
                tilt_angle_deg = np.rad2deg(data.joint("tilt").qpos[0])

                # Move the box toward its target position.
                t = (data.time - box_move_start_time) / box_move_duration

                if t >= 1.0:
                    box_start_pos = box_target_pos.copy()
                    box_target_pos = box_initial_pos + rand_pos_cube(1.0)
                    box_move_start_time = data.time
                    t = 0.0

                data.mocap_pos[mocap_id] = lerp(box_start_pos, box_target_pos, t)

                # Update the box's position in MuJoCo.
                body_id = model.geom_bodyid[target_id]
                mocap_id = model.body_mocapid[body_id]

                data.ctrl[pan_id] = np.deg2rad(pan_target)
                data.ctrl[tilt_id] = np.deg2rad(tilt_target)

                state_visualizer.update(
                    looking=looking,
                    ex=error_x,
                    ey=error_y,
                    pan_error_deg=pan_error_deg,
                    tilt_error_deg=tilt_error_deg,
                    K_p=K_P,
                    pan_deg=pan_angle_deg,
                    tilt_deg=tilt_angle_deg,
                    box_pos=data.mocap_pos[mocap_id].tolist()
                )

                mujoco.mj_step(model, data)
                viewer.sync()

                if step % render_every == 0:
                    renderer.update_scene(data, camera="ptz_cam")
                    _put_latest(frame_q, renderer.render().copy())
                step += 1
                time.sleep(max(0.0, dt - (time.monotonic() - t0)))
    finally:
        _put_latest(frame_q, None)
        proc.join(timeout=1)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)
        input_controller.stop()
        # input_visualizer.close()
        state_visualizer.close()


if __name__ == "__main__":
    main()