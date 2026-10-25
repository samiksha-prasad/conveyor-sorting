"""
The baseline colour detector, loaded straight from sorter.py.

sorter.py opens the Arduino serial port and the camera as soon as it is
imported, so it can't be imported in a simulation. Instead this module reads
sorter.py, extracts the source of its detect_color() function and runs
exactly that code. The simulation therefore evaluates sorter.py's function
itself, not a copy that could drift from it.
"""

import ast
import os

import cv2
import numpy as np

_SORTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sorter.py")


def _load_baseline():
    with open(_SORTER) as f:
        source = f.read()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "detect_color")
    namespace = {"cv2": cv2, "np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), _SORTER, "exec"), namespace)
    return namespace["detect_color"]


detect_colour = _load_baseline()
