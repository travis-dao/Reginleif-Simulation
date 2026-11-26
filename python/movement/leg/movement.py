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
    def solve(target: Vec3, coxa_offset: float, femur: float, tibia: float) -> Optional[Theta3]:
        x, y, z = target.x, target.y, target.z
        c_max = femur + tibia
        c_min = abs(femur - tibia)
 
        r = math.hypot(x, y) - coxa_offset
        femur_to_tip = math.hypot(z, r)
 
        if (femur_to_tip > c_max + EPSILON or femur_to_tip < c_min - EPSILON or femur_to_tip < EPSILON):
            return None
 
        a, b, c = femur, tibia, femur_to_tip
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

class FK:
    @staticmethod
    def solve(angles: Theta3, coxa_offset: float, femur: float,
              tibia: float) -> Vec3:
        """Inverse of IK.solve: IK-convention angles (deg) -> foot position (mm), leg frame.
        Pass the angles BEFORE IK.invert_angles / AFTER invert_angles has undone it."""
        coxa = angles.coxa * DEG_TO_RAD
        phi_f = angles.femur * DEG_TO_RAD                       # femur, from straight down toward +r
        phi_t = phi_f + angles.tibia * DEG_TO_RAD - math.pi     # tibia direction; tibia = knee interior angle

        r = femur * math.sin(phi_f) + tibia * math.sin(phi_t)
        z = -(femur * math.cos(phi_f) + tibia * math.cos(phi_t))

        rho = r + coxa_offset                                # horizontal distance from the leg origin
        return Vec3(rho * math.cos(coxa), rho * math.sin(coxa), z)

    @staticmethod
    def solve_servo(leg_id: int, servo_angles: Theta3, coxa_offset: float,
                    femur: float, tibia: float) -> Vec3:
        """From the angles actually sent to the servos (what Leg._move_leg produces)
        back to the leg-frame foot position, undoing the left-side invert and the right-side y flip."""
        right = leg_id == 1 or leg_id == 2
        a = servo_angles if right else IK.invert_angles(servo_angles)
        p = FK.solve(a, coxa_offset, femur, tibia)
        if right:
            p.y *= -1.0
        return p

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

        self.qadr: dict[str, int] = {}
        for leg in self.LEG_NAMES:
            for j in self.JOINTS:
                name = f"{leg}_{j}"                      # joints are named leg0_coxa, leg0_femur, ...
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid == -1:
                    raise KeyError(f"Joint '{name}' not found in model")
                self.qadr[name] = model.jnt_qposadr[jid]
 
    def set_angles(self, leg_idx: int, angles_deg: Theta3) -> None:
        """Write servo angles (deg) to the MuJoCo actuator controls (rad)."""
        leg = self.LEG_NAMES[leg_idx]
        self.data.ctrl[self.act_id[f"{leg}_coxa"]] = angles_deg.coxa * DEG_TO_RAD
        self.data.ctrl[self.act_id[f"{leg}_femur"]] = angles_deg.femur * DEG_TO_RAD
        self.data.ctrl[self.act_id[f"{leg}_tibia"]] = angles_deg.tibia * DEG_TO_RAD

        if self.debug is True:
            print(f"Leg {leg_idx} | coxa={angles_deg.coxa}, femur={angles_deg.femur}, tibia={angles_deg.tibia}")

    def get_angles(self, leg_idx: int) -> Theta3:
        """Measured servo angles (deg), same convention as set_angles."""
        leg = self.LEG_NAMES[leg_idx]
        q = self.data.qpos
        return Theta3(
            coxa=float(q[self.qadr[f"{leg}_coxa"]]) * RAD_TO_DEG,
            femur=float(q[self.qadr[f"{leg}_femur"]]) * RAD_TO_DEG,
            tibia=float(q[self.qadr[f"{leg}_tibia"]]) * RAD_TO_DEG,
        )