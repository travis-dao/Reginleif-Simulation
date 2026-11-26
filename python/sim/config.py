SRC = "models/body/xml/reginleif.xml"
DST = "models/body/xml/reginleif_sim.xml"

KP = 30.0
FORCE = 9.4

START_HEIGHT = 0.5 # drop height
STAND_HEIGHT = 0.07 # measure

# Terminate if the body tilts too far or falls too low.
FALL_UP_Z = 0.5
FALL_HEIGHT = 0.8 * STAND_HEIGHT  # 0.30 m

SETTLE_TIME_S = 0.1       # let the robot drop and stand on the kinematic gait before MPC engages
MPC_GAIN = 1.2            # mm of foot offset per N. Set 0.0 to run the pure gait (isolates MPC from gait problems)
SETTLE_VEL = 0.1          # ...and require the base to be (nearly) at rest

REST_S = 2.0              # stand still

LEG_NAMES = ("leg0", "leg1", "leg2", "leg3")
JOINTS = ("coxa", "femur", "tibia")

FOOT_BODIES = ("l0t", "l1t", "l2t", "l3t")
GROUND_GEOM = "floor"