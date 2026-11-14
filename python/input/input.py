
from dataclasses import dataclass
from input.keyboard import KeyboardInput
from math import hypot

@dataclass
class Input:
    x: float = 0.0
    y: float = 0.0


class InputController:
    MAX_INPUT_RATES = 15.0 # 1.0 -> 1 sec, 2 sec -> 0.5 sec
    def __init__(self):
        self.controls = KeyboardInput()
        self.input_ = Input()

    def update(self, dt_s):
        target_x, target_y = self.controls.get_axes()

        dx = target_x - self.input_.x
        dy = target_y - self.input_.y
        dist = hypot(dx, dy)
        max_change = self.MAX_INPUT_RATES * dt_s

        if dist > max_change and dist > 0.0:
            scale = max_change / dist
            dx *= scale
            dy *= scale

        self.input_.x += dx
        self.input_.y += dy

    def get_input(self) -> Input:
        return self.input_

    def stop(self):
        self.controls.stop()
