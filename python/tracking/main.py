# mjpython python/tracking/main.py

import time
import queue
import multiprocessing as mp

import sys
from pathlib import Path

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import numpy as np
import mujoco
import mujoco.viewer

from input.input import InputController
from input.visualizer import StateVisualizer

from tracking.util import rand_pos_cube, lerp
from tracking.tracking import Tracker


CAMERA_FPS = 30
CAMERA_NAME = "ptz_cam"      # one name for both the fovy lookup and the renderer


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


def camera_window(frame_q, event_q, cmd_q):
    """Detect the object and publish observations stamped with sim time.

    Hover the crosshair over the target (yellow circle), press SPACE to lock and
    start tracking, press SPACE again to unlock. SPACE works in this window or
    in the MuJoCo viewer.
    """
    import cv2
    import detection_ui as ui
    from obj_detection import ObjectDetector
    from acquisition import TargetLock, object_zone, crosshair_point, in_zone

    detector = ObjectDetector()
    ui.create_config_window()
    cv2.namedWindow(ui.VIEW_WINDOW, cv2.WINDOW_NORMAL)
    lock = TargetLock()
    show_mask = False

    while True:
        msg = frame_q.get()
        if msg is None:
            break
        sim_time, frame = msg                      # capture time on the sim clock

        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ui.read_config(detector.config)
        det = detector.detect(img)

        crosshair = crosshair_point(img.shape)
        zone = object_zone(det.bbox) if det.found else None
        hover = zone is not None and in_zone(crosshair, zone)

        key = cv2.waitKey(1) & 0xFF
        toggle = key == 32
        while True:                                # SPACE pressed in the viewer
            try:
                cmd_q.get_nowait()
                toggle = True
            except queue.Empty:
                break

        started, stopped = lock.update(hover, det.found, sim_time, toggle)

        h, w = img.shape[:2]
        _put_latest(event_q, {
            "detected": det.found,
            "center": det.center,
            "width": w,
            "height": h,
            "looking": lock.locked,
            "timestamp": sim_time,
        })

        if started:
            print("TARGET LOCKED — automatic tracking enabled")
        if stopped:
            print("TARGET UNLOCKED — manual control")

        cv2.imshow(ui.VIEW_WINDOW, ui.draw_overlay(img, det, lock.locked, hover, zone))
        if show_mask:
            cv2.imshow(ui.MASK_WINDOW, det.mask)

        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow(ui.MASK_WINDOW)
        if key == ord("p"):
            c = detector.config
            print(f"HSV lower={c.lower.tolist()} upper={c.upper.tolist()}")

    cv2.destroyAllWindows()


def main():
    input_controller = InputController()
    state_visualizer = StateVisualizer()

    model = mujoco.MjModel.from_xml_path("models/cam/pan_tilt.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    pan_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pan_servo")
    tilt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tilt_servo")
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "green_block")

    if cam_id < 0:
        raise RuntimeError(f"Camera '{CAMERA_NAME}' not found")
    if pan_id < 0 or tilt_id < 0:
        raise RuntimeError("Could not find pan_servo and tilt_servo actuators")
    mocap_id = model.body_mocapid[target_body_id]
    if mocap_id < 0:
        raise RuntimeError("green_block is not a mocap body")

    tracker = Tracker(model, data, fovy=float(model.cam_fovy[cam_id]))
    tracker.init(model, data, pan_id, tilt_id)

    frame_q = mp.Queue(maxsize=2)
    event_q = mp.Queue(maxsize=2)  # latest observation wins
    cmd_q = mp.Queue(maxsize=8)    # viewer -> camera process commands
    proc = mp.Process(target=camera_window, args=(frame_q, event_q, cmd_q), daemon=True)
    proc.start()

    dt = model.opt.timestep
    render_every = max(1, round(1 / CAMERA_FPS / dt))
    step = 0
    latest_observation = None      # cached between frames; the tracker relies on this

    box_initial_pos = data.mocap_pos[mocap_id].copy()
    box_start_pos = box_initial_pos.copy()
    box_target_pos = box_initial_pos.copy()
    box_move_start_time = data.time
    box_move_duration = 1.0

    def on_key(keycode):
        if keycode == ord("K"):
            tracker.use_kalman = not tracker.use_kalman
            print("Kalman filter:", "ON" if tracker.use_kalman else "OFF")
        elif keycode == 32:                      # SPACE: lock / unlock target
            _put_latest(cmd_q, "toggle")

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
            while viewer.is_running() and proc.is_alive():
                t0 = time.monotonic()
                input_controller.update(dt)
                cmd = input_controller.get_input()

                # Consume all queued observations and retain only the newest.
                while True:
                    try:
                        latest_observation = event_q.get_nowait()
                    except queue.Empty:
                        break

                tracker.update(dt, data.time, latest_observation, cmd)

                # Move the target box toward its next random position.
                t = (data.time - box_move_start_time) / box_move_duration
                if t >= 1.0:
                    box_start_pos = box_target_pos.copy()
                    box_target_pos = box_initial_pos + rand_pos_cube(1.0)
                    box_move_start_time = data.time
                    t = 0.0
                data.mocap_pos[mocap_id] = lerp(box_start_pos, box_target_pos, t)

                data.ctrl[pan_id] = np.deg2rad(tracker.pan())
                data.ctrl[tilt_id] = np.deg2rad(tracker.tilt())

                state_visualizer.update(
                    looking=tracker.looking,
                    pan_deg=np.rad2deg(data.joint("pan").qpos[0]),
                    tilt_deg=np.rad2deg(data.joint("tilt").qpos[0]),
                )

                mujoco.mj_step(model, data)
                viewer.sync()

                if step % render_every == 0:
                    renderer.update_scene(data, camera=CAMERA_NAME)
                    _put_latest(frame_q, (data.time, renderer.render().copy()))
                step += 1
                time.sleep(max(0.0, dt - (time.monotonic() - t0)))
    finally:
        _put_latest(frame_q, None)
        proc.join(timeout=1)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)
        input_controller.stop()
        state_visualizer.close()


if __name__ == "__main__":
    main()