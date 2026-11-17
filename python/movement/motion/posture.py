import math

class PostureShaper:
    def __init__(self, max_roll=math.radians(8), max_pitch=math.radians(6),
                 w_ref=0.5, v_ref=0.09, a_ref=0.15, tau=0.15,
                 turn_in_place_lean=0.5):
        self.roll = self.pitch = 0.0
        self.max_roll, self.max_pitch = max_roll, max_pitch
        self.w_ref, self.v_ref, self.a_ref = w_ref, v_ref, a_ref
        self.tau = tau
        self.tip = turn_in_place_lean      # fraction of full lean when v = 0

    def update(self, dt, v_mps, w_rad_s, a_mps2):
        clamp = lambda x: max(-1.0, min(1.0, x))

        # lean into the turn, scaled up with forward speed
        speed = clamp(abs(v_mps) / self.v_ref)
        gain = self.tip + (1.0 - self.tip) * speed
        roll_t = -self.max_roll * clamp(w_rad_s / self.w_ref) * gain

        # nose-down when accelerating, nose-up when braking
        pitch_t = self.max_pitch * clamp(a_mps2 / self.a_ref)

        a = 1.0 - math.exp(-dt / self.tau)
        self.roll += (roll_t - self.roll) * a
        self.pitch += (pitch_t - self.pitch) * a
        return self.roll, self.pitch