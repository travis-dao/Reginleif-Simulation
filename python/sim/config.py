SRC = "models/body/xml/reginleif.xml"
DST = "models/body/xml/reginleif_sim.xml"

KP = 30.0
FORCE = 9.4
START_HEIGHT = 0.5

HORIZON = 10
MPC_DT = 0.03
SETTLE_TIME_S = 2.0       # let the robot drop and stand on the kinematic gait before MPC engages
MPC_GAIN = 1.3            # mm of foot offset per N. Set 0.0 to run the pure gait (isolates MPC from gait problems)
SETTLE_VEL = 0.1          # ...and require the base to be (nearly) at rest

REST_S = 2.0              # stand still