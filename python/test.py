# random bs testing
# python python/test.py

from leg.leg import clamp_to_limits
from leg.leg_config import Theta3

print(clamp_to_limits(Theta3(180, 0, 180), 0))