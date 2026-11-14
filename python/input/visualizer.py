import json
import os
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from input.input import Input


def _child_python() -> str:
    """Interpreter for child processes. Prefer the venv's python3, since
    sys.executable may be the mjpython launcher (same main-thread problem)."""
    override = os.environ.get("VISUALIZER_PYTHON")
    if override:
        return override
    candidate = os.path.join(sys.prefix, "bin", "python3")
    return candidate if os.path.exists(candidate) else sys.executable


class PygameWindow:
    """Parent-side handle: one child process (and one window) per instance."""

    def __init__(self, kind: str, send_hz: float = 60.0):
        self.running = True
        self._period = 1.0 / send_hz
        self._last_send = 0.0
        self._proc = subprocess.Popen(
            [_child_python(), os.path.abspath(__file__), kind],
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def send(self, **values):
        if not self.running:
            return
        if self._proc.poll() is not None:      # window was closed
            self.running = False
            return

        now = time.time()
        if now - self._last_send < self._period:
            return
        self._last_send = now

        try:
            self._proc.stdin.write(json.dumps(values) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            self.running = False

    def close(self):
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=1.0)
        except Exception:
            self._proc.terminate()


class InputVisualizer(PygameWindow):
    """Joystick window. Same interface as before: update(command)."""

    def __init__(self):
        super().__init__("joystick")

    def update(self, command: "Input"):
        self.send(x=command.x, y=command.y)


class StateVisualizer(PygameWindow):
    """Second window: a text readout of whatever you pass to update()."""

    def __init__(self):
        super().__init__("state", send_hz=20.0)

    def update(self, **values):
        self.send(**values)


# ---------------------------------------------------------------------------
# Child process side: draw functions, one per window kind
# ---------------------------------------------------------------------------
def _draw_joystick(pg, screen, font, s):
    W, R = 320, 100
    cx, cy = W // 2, 135
    x, y = s.get("x", 0.0), s.get("y", 0.0)

    pg.draw.circle(screen, (90, 95, 110), (cx, cy), R, 2)
    pg.draw.line(screen, (65, 70, 85), (cx - R, cy), (cx + R, cy), 1)
    pg.draw.line(screen, (65, 70, 85), (cx, cy - R), (cx, cy + R), 1)

    # Positive Y is up, so invert screen coordinates.
    pg.draw.circle(screen, (227, 44, 27), (int(cx + x * R), int(cy - y * R)), 10)

    screen.blit(font.render(f"X: {x:+.2f}", True, (235, 235, 240)), (30, 275))
    screen.blit(font.render(f"Y: {y:+.2f}", True, (235, 235, 240)), (170, 275))


def _draw_state(pg, screen, font, s):
    for i, (k, v) in enumerate(s.items()):
        txt = f"{k}: {v:+.3f}" if isinstance(v, (int, float)) else f"{k}: {v}"
        screen.blit(font.render(txt, True, (235, 235, 240)), (20, 20 + 28 * i))


# kind -> (caption, (width, height), draw function)
WINDOWS = {
    "joystick": ("Input", (320, 360), _draw_joystick),
    "state":    ("State", (320, 240), _draw_state),
}


def _worker(kind: str):
    import pygame

    title, size, draw = WINDOWS[kind]
    state = {"alive": True, "data": {}}

    def reader():
        for line in sys.stdin:
            try:
                state["data"] = json.loads(line)
            except ValueError:
                pass
        state["alive"] = False                 # parent closed the pipe

    threading.Thread(target=reader, daemon=True).start()

    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption(title)
    font = pygame.font.Font(None, 26)
    clock = pygame.time.Clock()

    while state["alive"]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state["alive"] = False

        screen.fill((25, 28, 35))
        draw(pygame, screen, font, dict(state["data"]))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    _worker(sys.argv[1] if len(sys.argv) > 1 else "joystick")