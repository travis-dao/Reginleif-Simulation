import sys
from pathlib import Path

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PYTHON_DIR))

import mujoco
import numpy as np

import sim.config as cfg

class MuJuCoMap:
    """Same joint/actuator ordering as Env (leg0_coxa, leg0_femur, ...)."""
    def __init__(self, model):
        aids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{l}_{j}_act") for l in cfg.LEG_NAMES for j in cfg.JOINTS]
        jids = [int(model.actuator_trnid[a, 0]) for a in aids]
        self.aids = np.array(aids)
        self.q_adr = np.array([model.jnt_qposadr[j] for j in jids])
        self.v_adr = np.array([model.jnt_dofadr[j] for j in jids])
        self.limited = model.actuator_ctrllimited[self.aids].astype(bool)
        self.range = model.actuator_ctrlrange[self.aids].copy()
        self.foot_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in cfg.FOOT_BODIES]
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cfg.GROUND_GEOM)

    def contacts(self, model, data):
        c = np.zeros(4, dtype=bool)
        for i in range(data.ncon):
            g1, g2 = data.contact[i].geom1, data.contact[i].geom2
            if self.floor not in (g1, g2):
                continue
            other = g2 if g1 == self.floor else g1
            b = model.geom_bodyid[other]
            if b in self.foot_bodies:
                c[self.foot_bodies.index(b)] = True
        return c