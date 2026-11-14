# random bs testing
# python python/test.py

import time
from input.input import InputController
from input.visualizer import InputVisualizer

input_controller = InputController()
visualizer = InputVisualizer()

try:
    while visualizer.running:
        dt = 0.005  # Replace with measured elapsed time.

        input_controller.update(dt)
        command = input_controller.get_input()

        visualizer.update(command)

        # Update your MuJoCo simulation here.
finally:
    input_controller.stop()
    visualizer.close()