from functions import *
import optiview as ov
import numpy as np

v = ov.Visualizer()

rb = Rosenbrock()
v.plot(rb.fg, grad=True, ranges=rb.plot_ranges)
v.plot(rb.fg, grad=True, history=[np.array([-2, -4]),
                                  np.array([4, 5])],
       plot_history=False)


qp1d = QP1D()
v.plot(qp1d.fg, ranges=(-10, 10, 100))

qp2d = QP2D()
v.plot(qp2d.fg, ranges=qp2d.plot_ranges)
