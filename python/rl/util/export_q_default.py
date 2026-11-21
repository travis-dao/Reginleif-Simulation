# python python/rl/util/export_q_default.py

from pathlib import Path
import sys

import mujoco
import numpy as np

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PYTHON_DIR))

from sim.util.init_joint import init_joint
from sim.util.mujoco_robot_map import MuJuCoMap

XML_PATH = Path("models/body/xml/reginleif_sim.xml")
OUTPUT_PATH = Path("python/rl/data/q_default.npy")


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # Initialize the model to the default joint configuration.
    mujoco.mj_resetData(model, data)
    init_joint(model, data)
    mujoco.mj_forward(model, data)

    # Use the same actuator ordering as the RL environment.
    robot_map = MuJuCoMap(model)
    q_default = data.qpos[robot_map.q_adr].copy().astype(np.float32)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, q_default)

    print(f"Exported q_default: {OUTPUT_PATH}")
    print(f"Shape: {q_default.shape}")
    print("\nDefault joint positions:")

    for i, actuator_id in enumerate(robot_map.aids):
        actuator_name = model.actuator(actuator_id).name
        joint_id = model.actuator_trnid[actuator_id, 0]

        if joint_id >= 0:
            joint_name = model.joint(joint_id).name
            print(
                f"{i:2d}. {actuator_name:24s} "
                f"({joint_name}): {q_default[i]: .6f} rad "
                f"({np.rad2deg(q_default[i]): .2f} deg)"
            )
        else:
            print(f"{i:2d}. {actuator_name}: {q_default[i]: .6f}")

    print("\nSaved successfully.")


if __name__ == "__main__":
    main()