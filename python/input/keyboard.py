
from pynput import keyboard
from math import hypot


class KeyboardInput:
    def __init__(self):
        self._pressed = set()
        self._toggled = {}

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def _on_press(self, key):
        if hasattr(key, "char") and key.char:
            key_char = key.char.lower()

            # Toggle only once per physical key press.
            if key_char not in self._pressed:
                self._toggled[key_char] = not self._toggled.get(
                    key_char, False
                )

            self._pressed.add(key_char)

    def _on_release(self, key):
        if hasattr(key, "char") and key.char:
            self._pressed.discard(key.char.lower())

    def get_axes(self) -> tuple[float, float]:
        x = float("d" in self._pressed) - float("a" in self._pressed)
        y = float("w" in self._pressed) - float("s" in self._pressed)

        magnitude = hypot(x, y)
        if magnitude > 1.0:
            x /= magnitude
            y /= magnitude

        return x, y

    def get_key(self, key: str) -> bool:
        return self._toggled.get(key.lower(), False)

    def stop(self):
        self._listener.stop()