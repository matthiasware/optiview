import numpy as np


class Rosenbrock:
    def __init__(self):
        self.plot_ranges = ((-1.3, 1.2, 100),
                            (-1.7, 1.2, 100))

    def fg(self, x):
        f = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        g = np.array([400 * x[0]**3 - 400 * x[0] * x[1] + 2 * x[0] - 2,
                      200 * (x[1] - x[0]**2)])
        return f, g

    def h(self, x):
        return np.array([[1200 * x[0]**2 - 400 * x[1] + 2,
                          -400 * x[0]],
                         [-400 * x[0],
                          200]])


class QP1D:
    def __init__(self, a=1):
        self.plot_ranges = ((-10, 10, 100))
        self.a = a

    def fg(self, x):
        f = self.a * x**2
        g = self.a * 2 * x
        return f, g


class QP2D:
    def __init__(self, a=1, b=1):
        self.plot_ranges = ((-10, 10, 100), (-10, 10, 100))
        self.a = a
        self.b = b

    def fg(self, x):
        f = self.a * x[0]**2 + self.b * x[1]**2
        g = np.array([self.a * 2 * x[0],
                      self.b * 2 * x[1]])
        return f, g

    def h(self, x):
        return np.array([[self.a * 2, 0],
                         [0, self.b * 2]])
