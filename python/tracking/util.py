import random
import numpy as np

def rand_pos_cube(length: float) -> np.ndarray:
    half = length / 2.0
    return np.array([
        0,
        random.uniform(-half, half),
        random.uniform(-half, half),
    ], dtype=float)


def lerp(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
    return start + (end - start) * np.clip(t, 0.0, 1.0)