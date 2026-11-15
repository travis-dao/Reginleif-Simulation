import math
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
import argparse

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from input.smooth import smoothing


# Plot each piece separately to preserve discontinuities.
x1 = np.linspace(0.001001, 0.5, 1000, endpoint=False)
x2 = np.linspace(0.5, 0.8, 1000, endpoint=False)
x3 = np.linspace(0.8, 1.0, 1000, endpoint=False)
x4 = np.array([1.0])

pieces = [x1, x2, x3, x4]

for x in pieces:
    y = [smoothing(float(xi)) for xi in x]
    plt.plot(x, y, linewidth=2)

plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)

plt.xlim(0, 1)
plt.xlabel("x")
plt.ylabel("smoothing(x)")
plt.title("Smoothing Function")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()