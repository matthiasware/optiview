import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ranges can be (x,y,v) or ((x,y,u),)
# TODO:
# - plot partly:
#   1) funciton
#   2) if hsitory plot history on top
#   3) if animate, animate on top
# - unittests
# - ranges should be broadvasted e.g if (1,1,100) is given,
#   it should be used for all dimensions


class Visualizer:
    def __init__(self):
        self.NUM_1D_SR = 100
        self.NUM_2D_SR = 100
        self.NUM_3D_SR = 50
        self.OUTER_RANGE = 0.1
        self.COL_START = "g"
        self.COL_END = "r"
        self.OUT_OF_CONST = np.inf

    def plot(self, f, grad=True,
             ranges=None, history=None, plot_history=True,
             animate=False, d3=False, file_name=None, title=None,
             bounds=None, c=None, cbounds=None,
             orange=0.1, levels=40,
             points=None,  # (points, color, title)
             lines=None):  # (line, coler, title)
        if grad:
            def fnew(x):
                fv, _ = f(x)
                return fv
            self.f = fnew
        else:
            self.f = f
        self.OUTER_RANGE = orange
        if ranges is None:
            ranges = self._getRanges(history)
        # if ranges are (x, y, step) convert to ((x, y, z), )
        elif len(ranges) == 3:
            ranges = (ranges, )
        self.ranges = ranges
        self.dims = len(self.ranges)
        self.history = history
        self.plot_histroy = plot_history
        self.animate = animate
        self.d3 = d3
        self.file_name = file_name
        self.title = title
        self.c = c
        self.bounds = bounds
        self.cbounds = cbounds
        self.levels = levels
        self.points = points
        self.lines = lines
        plotter = self._determinePlotter()
        # if c:
            # self.plot_constraints()
        return plotter()

    def _getRanges(self, history):
        ranges = None
        if len(history) < 1:
            raise ValueError("Cannot infere ranges from empty history!")
        item = history[0]
        if isinstance(item, (int, float, np.number)):
            ranges = self._get1DRange(history)
        elif isinstance(item, np.ndarray):
            if item.size == 1:
                ranges = self._get1DRange(history)
            elif item.size == 2:
                ranges = self._get2DRange(history)
            else:
                raise ValueError("Cannot visualize a funciton"
                                 " of dimension {}.".format(item.size))
        elif isinstance(item, (list, tuple)):
            if len(item) == 1:
                ranges = self._get1DRange(history)
            elif len(item) == 2:
                ranges = self._get2DRange(history)
            else:
                raise ValueError("Cannot visualize a funciton"
                                 " of dimension {}.".format(item.size))
        else:
            raise ValueError("Unsupported history format!")
        return ranges

    def _get1DRange(self, history):
        x_min = min(history)
        x_max = max(history)
        diff = np.abs(x_max - x_min)
        x_min = x_min - self.OUTER_RANGE * diff
        x_max = x_max + self.OUTER_RANGE * diff
        return ((x_min, x_max, self.NUM_1D_SR),)

    def _get2DRange(self, history):
        xs, ys = zip(*history)
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        diff_x = np.abs(x_max - x_min)
        diff_y = np.abs(y_max - y_min)
        x_min = x_min - diff_x * self.OUTER_RANGE
        y_min = y_min - diff_y * self.OUTER_RANGE
        x_max = x_max + diff_x * self.OUTER_RANGE
        y_max = y_max + diff_y * self.OUTER_RANGE

        return ((x_min, x_max, self.NUM_2D_SR),
                (y_min, y_max, self.NUM_2D_SR))

    def _getDims(self, ranges):
        return len(ranges)

    def _determinePlotter(self):
        # (dims, animate, d3)
        d = {(1, True, True): self.plot2d_animate,
             (1, True, False): self.plot2d_animate,
             (1, False, False): self.plot2d,
             (1, False, True): self.plot2d,
             (2, True, True): self.plot3d_animate,
             (2, True, False): self.countour_animate,
             (2, False, True): self.plot3d,
             (2, False, False): self.contour}
        return d[(self.dims, self.animate, self.d3)]

    # def plot2d(self, f, ranges, history=None, grad=True):
    def plot2d(self):
        x = np.linspace(*self.ranges[0])
        y = [self.f(i) for i in x]
        plt.plot(x, y)
        if self.plot_histroy and self.history is not None:
            y = [self.f(i) for i in self.history]
            plt.plot(self.history, y)
        if self.title:
            plt.title(self.title)
        if self.bounds is not None:
            lb = self.bounds[0]
            ub = self.bounds[1]
            plt.axvline(x=lb, c="r")
            plt.axvline(x=ub, c="r")
        plt.show()

    # def plot2d_animate(self, f, ranges, history, grad=True):
    def plot2d_animate(self):
        fig = plt.figure()
        x = np.linspace(*self.ranges[0])
        y = [self.f(i) for i in x]
        plt.plot(x, y)

        if self.bounds is not None:
            lb = self.bounds[0]
            ub = self.bounds[1]
            plt.axvline(x=lb, c="r")
            plt.axvline(x=ub, c="r")

        def animate(frame_nr):
            x1 = self.history[frame_nr]
            x2 = self.history[frame_nr + 1]
            plt.plot([x1, x2], [self.f(x1), self.f(x2)])
        anim = animation.FuncAnimation(fig,
                                       animate,
                                       frames=range(len(self.history) - 1),
                                       repeat=False,
                                       interval=int(1e4) // len(self.history))
        # if self.title:
        # fig.title(self.title)
        plt.show()

    # def contour(self, f, ranges, history=None, grad=True):
    def contour(self):
        x = np.linspace(*self.ranges[0])
        y = np.linspace(*self.ranges[1])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                x = np.array([X[i][j], Y[i][j]])
                Z[i, j] = self.f(np.array(x))
                if self.c:
                    if not self.eval_c(x):
                        Z[i, j] = self.OUT_OF_CONST
        cp = plt.contourf(X, Y, Z, self.levels)
        plt.colorbar(cp)

        if self.history is not None and self.plot_histroy:
            x, y = zip(*self.history)  # in matplotlib no generators!
            plt.plot(x, y)
            plt.scatter(x[1:-1], y[1:-1], c="b")
            plt.scatter(x[0], y[0], c=self.COL_START)
            plt.scatter(x[-1], y[-1], c=self.COL_END)
        if self.title:
            plt.title(self.title)
        if self.bounds is not None:
            lb = self.bounds[0]
            ub = self.bounds[1]
            plt.axhline(y=lb[1], color='r', linestyle='-')
            plt.axhline(y=ub[1], color='r', linestyle='-')
            plt.axvline(x=ub[0], color='r', linestyle='-')
            plt.axvline(x=lb[0], color='r', linestyle='-')
        if self.points is not None:
            x, y = zip(*self.points)
            plt.scatter(x, y, c="deeppink", alpha=0.3, s=200)
        plt.show()

    def plot_constraints(self):
        x = np.linspace(*self.ranges[0])
        y = np.linspace(*self.ranges[1])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i, j] = self.eval_c(np.array([X[i][j], Y[i][j]]))
        cp = plt.contourf(X, Y, Z, self.levels)
        plt.colorbar(cp)

        if self.history is not None and self.plot_histroy:
            x, y = zip(*self.history)  # in matplotlib no generators!
            plt.plot(x, y)
            plt.scatter(x[1:-1], y[1:-1], c="b")
            plt.scatter(x[0], y[0], c=self.COL_START)
            plt.scatter(x[-1], y[-1], c=self.COL_END)
        if self.title:
            plt.title(self.title)
        plt.show()

    # def countour_animate(self, f, ranges, history, grad=True):
    def countour_animate(self):
        x = np.linspace(*self.ranges[0])
        y = np.linspace(*self.ranges[1])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                x = np.array([X[i][j], Y[i][j]])
                Z[i, j] = self.f(x)
                if self.c:
                    if not self.eval_c(x):
                        Z[i, j] = self.OUT_OF_CONST
        fig = plt.figure()
        cp = plt.contourf(X, Y, Z, self.levels)
        plt.colorbar(cp)

        x, y = zip(*self.history)
        plt.scatter(x[0], y[0], c=self.COL_START)
        plt.scatter(x[-1], y[-1], c=self.COL_END)

        if self.bounds is not None:
            lb = self.bounds[0]
            ub = self.bounds[1]
            plt.axhline(y=lb[1], color='r', linestyle='-')
            plt.axhline(y=ub[1], color='r', linestyle='-')
            plt.axvline(x=ub[0], color='r', linestyle='-')
            plt.axvline(x=lb[0], color='r', linestyle='-')

        def animate(frame_nr):
            plt.plot([x[frame_nr], x[frame_nr + 1]],
                     [y[frame_nr], y[frame_nr + 1]])
        anim = animation.FuncAnimation(fig,
                                       animate,
                                       frames=range(len(self.history) - 1),
                                       repeat=False,
                                       interval=int(1e4) // len(self.history))
        # if self.title:
        # fig.title(self.title)
        plt.show()
        # anim.save("x.mp4")

    # def plot3d(self, f, ranges, history=None, grad=True):
    def plot3d(self):
        x = np.linspace(*self.ranges[0])
        y = np.linspace(*self.ranges[1])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                x = np.array([X[i][j], Y[i][j]])
                Z[i, j] = self.f(x)
                if self.c:
                    if not self.eval_c(x):
                        Z[i, j] = self.OUT_OF_CONST
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0,
                        antialiased=False, alpha=0.3)
        if self.history is not None and self.plot_histroy:
            x, y, z = zip(*[(i[0], i[1], self.f(i)) for i in self.history])
            plt.plot(x, y, z)
            ax.scatter(x[0], y[0], z[0], c=self.COL_START)
            ax.scatter(x[-1], y[-1], z[-1], c=self.COL_END)
        # if self.bounds is not None:
            # lb = self.bounds[0]
            # ub = self.bounds[1]

            # works
            # yy, zz = np.meshgrid(y, range(int(np.max(Z))))
            # xx = yy * 0
            # ax.plot_surface(xx + lb[0], yy, zz)
            # ax.plot_surface(xx + ub[0], yy, zz)
        plt.show()
        # if self.title:
            # ax.title(self.title)
        return Z

    def eval_c(self, x):
        c = self.c(x)
        cu = c <= self.cbounds[1]
        cl = self.cbounds[0] <= c
        return cu.all() and cl.all()

    # def plot3d_animate(self, f, ranges, history, grad=True):
    def plot3d_animate(self):
        x = np.linspace(*self.ranges[0])
        y = np.linspace(*self.ranges[1])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                x = np.array([X[i][j], Y[i][j]])
                Z[i, j] = self.f(x)
                if self.c:
                    if not self.eval_c(x):
                        Z[i, j] = self.OUT_OF_CONST
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0,
                        antialiased=False, alpha=0.3)
        x, y, z = zip(*[(i[0], i[1], self.f(i)) for i in self.history])
        # plt.plot(x, y, z)
        ax.scatter(x[0], y[0], z[0], c=self.COL_START)
        ax.scatter(x[-1], y[-1], z[-1], c=self.COL_END)

        def animate(frame_nr):
            plt.plot([x[frame_nr], x[frame_nr + 1]],
                     [y[frame_nr], y[frame_nr + 1]],
                     [z[frame_nr], z[frame_nr + 1]])
        anim = animation.FuncAnimation(fig,
                                       animate,
                                       frames=range(len(self.history) - 1),
                                       repeat=False,
                                       interval=int(1e4) // len(self.history))
        # if self.title:
        # ax.title(self.title)
        plt.show()
        # anim.save("x.mp4")
