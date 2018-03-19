from optiview import Visualizer
import unittest
import numpy as np


class TestVisualizer(unittest.TestCase):
    def test_determinePlotter(self):
        v = Visualizer()
        v.dims, v.animate, v.d3 = 1, True, True
        self.assertEqual(v._determinePlotter(), v.plot2d_animate)
        v.dims, v.animate, v.d3 = 1, True, False
        self.assertEqual(v._determinePlotter(), v.plot2d_animate)
        v.dims, v.animate, v.d3 = 1, False, True
        self.assertEqual(v._determinePlotter(), v.plot2d)
        v.dims, v.animate, v.d3 = 1, False, False
        self.assertEqual(v._determinePlotter(), v.plot2d)
        v.dims, v.animate, v.d3 = 2, True, True
        self.assertEqual(v._determinePlotter(), v.plot3d_animate)
        v.dims, v.animate, v.d3 = 2, True, False
        self.assertEqual(v._determinePlotter(), v.countour_animate)
        v.dims, v.animate, v.d3 = 2, False, True
        self.assertEqual(v._determinePlotter(), v.plot3d)
        v.dims, v.animate, v.d3 = 2, False, False
        self.assertEqual(v._determinePlotter(), v.contour)

    def test_getRanges(self):
        v = Visualizer()
        v.OUTER_RANGE = 0.0
        # 1D RANGES
        h = [np.array(5), np.array(0.1), np.array(-2)]
        self.assertEqual(v._getRanges(h), ((-2, 5, v.NUM_1D_SR),))
        h = [1, 20, .5, 4.3]
        self.assertEqual(v._getRanges(h), ((.5, 20, v.NUM_1D_SR),))
        h = [-.1, .2, .3, .4]
        self.assertEqual(v._getRanges(h), ((-.1, .4, v.NUM_1D_SR),))
        h = np.array([-10, 6, -3, 2, 0])
        self.assertEqual(v._getRanges(h), ((-10, 6, 100), ))
        # 2D RANGES
        h = [np.array([1, 2]), np.array([0, 6]), np.array([4.0, 5.1])]
        self.assertEqual(v._getRanges(h), ((0.0, 4.0, v.NUM_2D_SR),
                                           (2, 6, v.NUM_2D_SR)))
        h = [[1, 2], [0, 6], [4.0, 5.1]]
        self.assertEqual(v._getRanges(h), ((0.0, 4.0, v.NUM_2D_SR),
                                           (2, 6, v.NUM_2D_SR)))

    def test_plot2d(self):
        def qpg(x):
            return x**2, 2 * x

        def qp(x):
            return x**2

        v = Visualizer()
        v.OUTER_RANGE = 0.0

        history = np.array([-10, 6, -3, 2, 0])
        ranges = (-10, 10, 100)

        v.plot(f=qpg, history=history, title="Ranges from history")
        v.plot(f=qp, grad=False, history=history, ranges=ranges,
               title="History + Range")
        v.plot(f=qp, grad=False, history=history, plot_history=False,
               title="Range fom history, plot_history=False")
        v.plot(f=qp, grad=False, ranges=(ranges,),
               title="ranges without history")
        v.plot(f=qp, grad=False, history=history,
               animate=True, title="Animate")

    def test_contour(self):
        def qp(x):
            return sum(i**2 for i in x)

        v = Visualizer()
        v.OUTER_RANGE = 0.0

        history = [np.array([-10, 5]), np.array([4, -8]),
                   np.array([-7, 13]), np.array([0, 0])]
        ranges = ((-100, 100, 100), (-20, 20, 100))
        v.plot(f=qp, grad=False, history=history,
               animate=False, title="Contour")
        v.plot(f=qp, grad=False, history=history,
               plot_history=False, title="Contour no history")
        v.plot(f=qp, grad=False, history=history, ranges=ranges,
               title="Contour Ranges and History")
        v.plot(f=qp, grad=False, history=history,
               animate=True, title="Contour animate")

    def test_plot3d(self):
        def qp(x):
            return sum(i**2 for i in x)

        v = Visualizer()
        v.OUTER_RANGE = 0.0

        history = [np.array([-10, 5]), np.array([4, -8]),
                   np.array([-7, 13]), np.array([0, 0])]
        ranges = ((-100, 100, 100), (-20, 20, 100))
        v.plot(f=qp, grad=False, history=history,
               animate=False, title="3D", d3=True)
        v.plot(f=qp, grad=False, history=history,
               plot_history=False, title="3D from ranges", d3=True)
        v.plot(f=qp, grad=False, history=history, ranges=ranges,
               title="3D Ranges and History", d3=True)
        v.plot(f=qp, grad=False, history=history,
               animate=True, title="3D animate", d3=True)


if __name__ == "__main__":
    unittest.main()
