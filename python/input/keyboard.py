
from pynput import keyboard
from math import hypot


class KeyboardInput:
    def __init__(self):
        self._pressed = set()

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def _on_press(self, key):
        if hasattr(key, "char") and key.char:
            self._pressed.add(key.char.lower())

    def _on_release(self, key):
        if hasattr(key, "char") and key.char:
            self._pressed.discard(key.char.lower())

    def get_axes(self) -> tuple[float, float]:
        x = float("d" in self._pressed) - float("a" in self._pressed)
        y = float("w" in self._pressed) - float("s" in self._pressed)

        # Normalize diagonals so the magnitude never exceeds 1.
        magnitude = hypot(x, y)
        if magnitude > 1.0:
            x /= magnitude
            y /= magnitude

        return x, y

    def stop(self):
        self._listener.stop()