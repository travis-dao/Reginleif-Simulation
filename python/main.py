# make sure xml mesh paths are like "file="../assembly/meshes/*"
# source .venv/bin/activate
# mjpython python/main.py

import time
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("assembly/xml/reginleif.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        data.ctrl[:] = 0.5          # your controller goes here
        mujoco.mj_step(model, data)

        viewer.sync()               # push the new state to the window

        # keep real-time pace
        sleep = model.opt.timestep - (time.time() - step_start)
        if sleep > 0:
            time.sleep(sleep)