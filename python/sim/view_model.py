from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

import sys
from pathlib import Path

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from movement.leg.leg_controller import LegController
from sim.util.init_joint import init_joint

XML_PATH = Path("models/body/xml/reginleif_sim.xml")

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

# Reset simulation to the model's initial state.
mujoco.mj_resetData(model, data)

init_joint(model, data)

# model.opt.gravity[:] = [0.0, 0.0, 0.0]

mujoco.mj_forward(model, data)

# lc = LegController(model, data)

printed_joint_angles = False

print(f"Model has {model.njnt} joints", flush=True)
print("Joint types:", [(model.joint(j).name, int(model.jnt_type[j])) for j in range(model.njnt)], flush=True)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # lc.update(model.opt.timestep, None)

        mujoco.mj_step(model, data)
        viewer.sync()

        if data.time >= 1.0 and not printed_joint_angles:
            print(f"\n=== Joint Angles at t={data.time:.2f}s ===", flush=True)

            hinge_count = 0

            for j in range(model.njnt):
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                    name = model.joint(j).name
                    qpos_addr = model.jnt_qposadr[j]
                    angle_rad = data.qpos[qpos_addr]

                    print(f"{name}: {angle_rad:.4f} rad ({np.rad2deg(angle_rad):.2f} deg)", flush=True)
                    hinge_count += 1

            if hinge_count == 0:
                print("No hinge joints found.", flush=True)

            printed_joint_angles = True