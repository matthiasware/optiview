from functions import *
from visualize import Visualizer
import numpy as np

v = Visualizer()
rb = Rosenbrock()
v.plot(rb.fg, grad=True, ranges=rb.plot_ranges)
v.plot(rb.fg, grad=True, history=[np.array([-2, -4]),
                                  np.array([4, 5])],
       plot_history=False)
