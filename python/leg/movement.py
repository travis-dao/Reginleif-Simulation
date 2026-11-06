import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Union
 
import mujoco
from .leg_config import Vec3, Theta3
 
EPSILON = 1e-6
RAD_TO_DEG = 180.0 / math.pi
DEG_TO_RAD = math.pi / 180.0
 
def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
 
 
# --------------------------------------------------------------------------
# inverse kinematics (pure math, no simulator)
# --------------------------------------------------------------------------
class IK:
    @staticmethod
    def solve(target: Vec3, body_coxa_x_mm: float, femur_length_mm: float, tibia_length_mm: float, c_min: float, c_max: float) -> Optional[Theta3]:
        x, y, z = target.x, target.y, target.z
 
        r = math.hypot(x, y) - body_coxa_x_mm
        femur_to_tip = math.hypot(z, r)
 
        if (femur_to_tip > c_max + EPSILON
                or femur_to_tip < c_min - EPSILON
                or femur_to_tip < EPSILON):
            return None
 
        a, b, c = femur_length_mm, tibia_length_mm, femur_to_tip
        cos2 = clamp((a * a + c * c - b * b) / (2 * a * c), -1.0, 1.0)
        cos3 = clamp((a * a + b * b - c * c) / (2 * a * b), -1.0, 1.0)
 
        return Theta3(
            coxa=math.atan2(y, x) * RAD_TO_DEG,
            femur=(math.atan2(r, -z) + math.acos(cos2)) * RAD_TO_DEG,
            tibia=math.acos(cos3) * RAD_TO_DEG,
        )
 
    @staticmethod
    def invert_angles(a: Theta3) -> Theta3:
        """Flip for reverse-mounted servos."""
        return Theta3(180.0 - a.coxa, 180.0 - a.femur, 180.0 - a.tibia)


# --------------------------------------------------------------------------
# MuJoCo interface
# --------------------------------------------------------------------------
class LegDriver:
    LEG_NAMES = ("leg0", "leg1", "leg2", "leg3")
    JOINTS = ("coxa", "femur", "tibia")
 
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, debug: bool = False):
        self.model = model
        self.data = data
        self.debug = debug
 
        self.act_id: dict[str, int] = {}
        for leg in self.LEG_NAMES:
            for j in self.JOINTS:
                name = f"{leg}_{j}_act"
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if aid == -1:
                    raise KeyError(f"Actuator '{name}' not found in model")
                self.act_id[f"{leg}_{j}"] = aid
 
    def set_angles(self, leg_idx: int, angles_deg: Theta3) -> None:
        """Write servo angles (deg) to the MuJoCo actuator controls (rad)."""
        leg = self.LEG_NAMES[leg_idx]
        self.data.ctrl[self.act_id[f"{leg}_coxa"]] = angles_deg.coxa * DEG_TO_RAD
        self.data.ctrl[self.act_id[f"{leg}_femur"]] = angles_deg.femur * DEG_TO_RAD
        self.data.ctrl[self.act_id[f"{leg}_tibia"]] = angles_deg.tibia * DEG_TO_RAD

        if self.debug is True:
            print(f"Leg {leg_idx} | coxa={angles_deg.coxa}, femur={angles_deg.femur}, tibia={angles_deg.tibia}")