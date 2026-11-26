from typing import Callable
import numpy as np

from .leg_config import BODY_COXA_X_MM, FEMUR_LENGTH_MM, TIBIA_LENGTH_MM, Theta3
from .leg_controller import leg_to_body_frame
from .movement import FK

MM = 1e-3

class MeasuredFootKinematics:
    """Foot positions in the body frame [m] from measured servo angles, via FK.
    Same frame and ordering (FL, FR, BR, BL) as LegController.foot_positions_body(), but built from
    where the servos actually are, not from what was commanded."""

    def __init__(self, read_servo_angles: Callable[[int], Theta3]):
        self.read = read_servo_angles                      # leg index -> Theta3, servo convention, degrees

    def __call__(self) -> np.ndarray:
        out = np.empty((4, 3))
        for i in range(4):
            p_lf = FK.solve_servo(i, self.read(i), BODY_COXA_X_MM, FEMUR_LENGTH_MM, TIBIA_LENGTH_MM)
            p_bf = leg_to_body_frame(i, p_lf)
            out[i] = (p_bf.x * MM, p_bf.y * MM, p_bf.z * MM)
        return out