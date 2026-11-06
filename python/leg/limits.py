from dataclasses import dataclass


@dataclass(frozen=True)
class ServoLimit:
    min_deg: float
    max_deg: float

    def clamp(self, angle: float) -> float:
        return max(self.min_deg, min(angle, self.max_deg))


@dataclass(frozen=True)
class LegServoLimits:
    coxa: ServoLimit
    femur: ServoLimit
    tibia: ServoLimit


class ServoConfig:
    limits = [
        LegServoLimits(  # Front Left (0)
            coxa=ServoLimit(45.0, 180.0),
            femur=ServoLimit(40.0, 135.0),
            tibia=ServoLimit(0.0, 135.0),
        ),
        LegServoLimits(  # Front Right (1)
            coxa=ServoLimit(0.0, 135.0),
            femur=ServoLimit(45.0, 140.0),
            tibia=ServoLimit(45.0, 180.0),
        ),
        LegServoLimits(  # Back Right (2)
            coxa=ServoLimit(45.0, 170.0),
            femur=ServoLimit(45.0, 140.0),
            tibia=ServoLimit(45.0, 180.0),
        ),
        LegServoLimits(  # Back Left (3)
            coxa=ServoLimit(10.0, 135.0),
            femur=ServoLimit(40.0, 135.0),
            tibia=ServoLimit(0.0, 135.0),
        ),
    ]