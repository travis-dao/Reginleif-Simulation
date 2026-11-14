import os
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from input.input import Input


def _child_python() -> str:
    """Interpreter for the child process. Prefer the venv's python3, since
    sys.executable may be the mjpython launcher (which has the same
    main-thread problem)."""
    override = os.environ.get("VISUALIZER_PYTHON")
    if override:
        return override
    candidate = os.path.join(sys.prefix, "bin", "python3")
    return candidate if os.path.exists(candidate) else sys.executable


class InputVisualizer:
    """Parent-side handle. Spawns the pygame window in a separate process."""

    SEND_HZ = 60

    def __init__(self):
        self.running = True
        self._last_send = 0.0
        self._proc = subprocess.Popen(
            [_child_python(), os.path.abspath(__file__)],
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def update(self, command: "Input"):
        if not self.running:
            return
        if self._proc.poll() is not None:      # window was closed
            self.running = False
            return

        now = time.time()
        if now - self._last_send < 1.0 / self.SEND_HZ:
            return
        self._last_send = now

        try:
            self._proc.stdin.write(f"{command.x} {command.y}\n")
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


# ---------------------------------------------------------------------------
# Child process: owns the pygame window (runs on that process's main thread)
# ---------------------------------------------------------------------------
def _worker():
    import pygame

    WIDTH, HEIGHT, RADIUS = 320, 360, 100

    state = {"x": 0.0, "y": 0.0, "alive": True}

    def reader():
        for line in sys.stdin:
            try:
                x, y = line.split()
                state["x"], state["y"] = float(x), float(y)
            except ValueError:
                pass
        state["alive"] = False                 # parent closed the pipe

    threading.Thread(target=reader, daemon=True).start()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Input")
    font = pygame.font.Font(None, 26)
    clock = pygame.time.Clock()

    while state["alive"]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state["alive"] = False

        x, y = state["x"], state["y"]
        screen.fill((25, 28, 35))
        cx, cy = WIDTH // 2, 135

        pygame.draw.circle(screen, (90, 95, 110), (cx, cy), RADIUS, 2)
        pygame.draw.line(screen, (65, 70, 85), (cx - RADIUS, cy), (cx + RADIUS, cy), 1)
        pygame.draw.line(screen, (65, 70, 85), (cx, cy - RADIUS), (cx, cy + RADIUS), 1)

        # Positive Y is up, so invert screen coordinates.
        pygame.draw.circle(
            screen, (227, 44, 27),
            (int(cx + x * RADIUS), int(cy - y * RADIUS)), 10
        )

        screen.blit(font.render(f"X: {x:+.2f}", True, (235, 235, 240)), (30, 275))
        screen.blit(font.render(f"Y: {y:+.2f}", True, (235, 235, 240)), (170, 275))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    _worker()