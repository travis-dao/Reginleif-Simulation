# mjpython python/main.py

import time
import numpy as np
import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET

from leg.leg_controller import LegController
from mpc import State, Input, Model, DiscreteModel, MPCProblem, MPCResult, MPCConfig, MPC
from state import read_state

SRC = "assembly/xml/reginleif.xml"        # your original file
DST = "assembly/xml/reginleif_sim.xml"     # must be saved in the same folder as SRC

KP = 30.0                   # servo stiffness (tune)
FORCE = 9.4                 # max joint torque in N*m (set to your servo spec)
START_HEIGHT = 0.5         # drop height of the base, adjust to your geometry

# ---------- build the simulation XML ----------
tree = ET.parse(SRC)
root = tree.getroot()

ET.SubElement(root, "option", timestep="0.002")

wb = root.find("worldbody")
base = ET.Element("body", name="base", pos=f"0 0 {START_HEIGHT}")
ET.SubElement(base, "freejoint", name="root")
for child in list(wb):          # move all base geoms and leg bodies under "base"
    wb.remove(child)
    base.append(child)
wb.append(base)

# collisions: disable all mesh geoms, then enable only the ankles + floor
for g in base.iter("geom"):
    g.set("contype", "0")
    g.set("conaffinity", "0")
    if "ankle" in g.get("mesh", ""):
        g.set("contype", "1")
        g.set("conaffinity", "1")
        g.set("friction", "0.7 0.005 0.0001")

# Add this inside <asset> before the floor geom is created
asset = root.find("asset")
ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
              rgb1="1 1 1", rgb2="0.9 0.9 0.9", width="300", height="300")
ET.SubElement(asset, "material", name="floor_mat", texture="grid",
              texrepeat="5 5", reflectance="0.1")

ET.SubElement(wb, "geom", name="floor", type="plane", size="2 2 0.1",
              contype="1", conaffinity="1", material="floor_mat")

# joint damping + position actuators
act = ET.SubElement(root, "actuator")
for j in base.iter("joint"):
    if j.get("name") == "root":
        continue
    j.set("damping", "0.5")
    ET.SubElement(act, "position", name=j.get("name") + "_act", joint=j.get("name"), kp=str(KP), forcerange=f"-{FORCE} {FORCE}")

tree.write(DST)

model = mujoco.MjModel.from_xml_path(DST)
data = mujoco.MjData(model)

controller = LegController(model, data)
controller.init()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # update gaits

        # read current model state

        # mpc.solve

        # update foot position
        controller.update(0.001, None)

        mujoco.mj_step(model, data)
        viewer.sync()
        sleep = model.opt.timestep - (time.time() - step_start)
        if sleep > 0:
            time.sleep(sleep)