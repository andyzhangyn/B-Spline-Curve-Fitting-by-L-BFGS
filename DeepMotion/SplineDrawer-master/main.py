#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Chi Zhang (chizhang@deepmotion.ai)


import copy
import sys
import numpy as np
from PyQt4 import QtGui, QtCore


class LaneLineSpline(object):

    def __init__(self):
        # _vertices is a (n+1)x2 array representing the n+1 spline knots.
        self._vertices = np.zeros((0, 2), np.float32)

        # Each list item is a 4x2 numpy array
        # representing 4 bezier control points.
        self._segment_wise_bezier_control_points = []

        # Conclusion: using float will cause nemerical issue when the curve
        # is long.
        self.is_use_double = False

    def size(self):
        return len(self._vertices)

    def empty(self):
        return self.size() == 0

    def add_vertex(self, coord):
        coord = np.array([coord.x(), coord.y()], np.float32)
        if self.empty() or \
                np.linalg.norm(self._vertices[-1, :] - coord) >= 1.0:
            self._vertices = np.vstack((self._vertices, coord))
            self.recompute_bezier_control_points()

    def remove_last_vertex(self):
        if not self.empty():
            self._vertices = self._vertices[:-1, :]
            self.recompute_bezier_control_points()

    def get_bezier_segments(self):
        return self._segment_wise_bezier_control_points

    def get_spline_control_points(self):
        return self._vertices

    def toggle_float_accuracy(self):
        self.is_use_double = not self.is_use_double

    def recompute_bezier_control_points(self):
        self._segment_wise_bezier_control_points = []
        section_wise_x0123s = []
        section_wise_y0123s = []

        if len(self._vertices) <= 1:
            return

        for column_idx in [0, 1]:
            section_wise_abcds, section_wise_t0t1s = \
                self._compute_cubic_spline_coeffs(
                    self._vertices[:, column_idx])

            section_wise_p0123s = []
            for abcd, t0t1 in zip(section_wise_abcds, section_wise_t0t1s):
                t0, dt = t0t1[0], t0t1[1] - t0t1[0]
                p0123 = self._from_spline_abcd_to_bezier_p0123(abcd, t0, dt)
                section_wise_p0123s.append(p0123)

            if column_idx == 0:
                section_wise_x0123s = section_wise_p0123s
            else:
                section_wise_y0123s = section_wise_p0123s

        for x0123, y0123 in zip(section_wise_x0123s, section_wise_y0123s):
            self._segment_wise_bezier_control_points.append(
                np.hstack((x0123.reshape((4, 1)), y0123.reshape((4, 1)))))

    def _compute_cubic_spline_coeffs(self, x):
        n = len(self._vertices) - 1
        A = np.zeros((4*n, 4*n), np.float64)
        b = np.zeros((4*n,), np.float64)

        if self.is_use_double:
            lengths = np.linalg.norm(
                self._vertices[1:, :] - self._vertices[:-1, :], axis=1).astype(
                dtype=np.float64)
        else:
            lengths = np.linalg.norm(
                self._vertices[1:, :] - self._vertices[:-1, :], axis=1)

        t = [0.0] + \
            [np.sum(lengths[:i]) / np.sum(lengths) for i in range(1, n+1)]

        # n value constraints from left endpoints.
        for i in range(n):
            A[i, 4*i: 4*i+4] = [t[i]*t[i]*t[i], t[i]*t[i], t[i], 1.0]
            b[i] = x[i]
        # n value constraints from right endpoints.
        for i in range(n):
            A[n+i, 4*i: 4*i+4] = \
                [t[i+1]*t[i+1]*t[i+1], t[i+1]*t[i+1], t[i+1], 1.0]
            b[n+i] = x[i+1]
        # (n - 1) first order derivative constraints.
        for i in range(n - 1):
            A[2*n+i, 4*i: 4*i+4] = \
                [3*t[i+1]*t[i+1], 2*t[i+1], 1.0, 0.0]
            A[2*n+i, 4*(i+1): 4*(i+1)+4] = \
                [-3*t[i+1]*t[i+1], -2*t[i+1], -1.0, -0.0]
        # (n - 1) second order derivative constraints.
        for i in range(n - 1):
            A[3*n-1+i, 4*i: 4*i+4] = [6*t[i+1], 2.0, 0.0, 0.0]
            A[3*n-1+i, 4*(i+1): 4*(i+1)+4] = [-6*t[i+1], -2.0, 0.0, 0.0]
        # Two second order derivative constraints from two out-most endpoints.
        # 2*b_0 = 0; 6*a_{n-1} + 2*b_{n-1} = 0
        A[-2, 4*0+1] = 2.0
        A[-1, 4*(n-1)+0] = 6.0
        A[-1, 4*(n-1)+1] = 2.0

        abcds = np.linalg.solve(A, b)
        return [abcds[4*i: 4*i+4] for i in range(n)], \
               [(t[i], t[i+1]) for i in range(n)]

    @staticmethod
    def _from_spline_abcd_to_bezier_p0123(abcd, t0, dt):
        a, b, c, d = abcd[0], abcd[1], abcd[2], abcd[3]
        A = a*dt*dt*dt
        B = dt*dt*(b + 3*a*t0)
        C = dt*(3*a*t0*t0 + 2*b*t0 + c)
        D = a*t0*t0*t0 + b*t0*t0 + c*t0 + d
        p0 = D
        p1 = (C + 3*p0) / 3.0
        p2 = (B - 3*p0 + 6*p1) / 3.0
        p3 = A + p0 - 3*p1 + 3*p2
        return np.array([p0, p1, p2, p3])


class MainWindow(QtGui.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.lane_line = LaneLineSpline()
        self.mouse_pos = QtCore.QPointF()

        self.setMouseTracking(True)
        self.showMaximized()

    def mousePressEvent(self, event):
        self.update_mouse(event)
        if event.button() == QtCore.Qt.LeftButton:
            self.lane_line.add_vertex(event.posF())
            self.update()
        if event.button() == QtCore.Qt.RightButton:
            self.lane_line.remove_last_vertex()
            self.update()

    def mouseMoveEvent(self, event):
        self.update_mouse(event)
        self.update()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self.lane_line.toggle_float_accuracy()
            self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        self.draw_lane_line(qp, self.lane_line)
        qp.end()

    def update_mouse(self, event):
        self.mouse_pos = event.posF()

    def draw_lane_line(self, qp, lane_line):
        lane_line = copy.deepcopy(lane_line)
        lane_line.add_vertex(self.mouse_pos)

        qp.setRenderHint(QtGui.QPainter.Antialiasing)

        # Draw spline curve.
        path = QtGui.QPainterPath()
        bezier_segments = lane_line.get_bezier_segments()
        for bezier_segment in bezier_segments:
            bezier_control_points = [
                QtCore.QPointF(bezier_segment[i, 0], bezier_segment[i, 1])
                for i in range(4)]

            path.moveTo(bezier_control_points[0])
            path.cubicTo(bezier_control_points[1],
                         bezier_control_points[2],
                         bezier_control_points[3])

        qp.setPen(QtGui.QPen(
            QtGui.QBrush(QtGui.QColor(255, 0, 0), QtCore.Qt.SolidPattern),
            1.0, QtCore.Qt.SolidLine))
        qp.drawPath(path)

        # Draw_control_points.
        spline_control_points = lane_line.get_spline_control_points()
        spline_control_points = [QtCore.QPointF(vertex[0], vertex[1])
                                 for vertex in spline_control_points]

        if len(spline_control_points) > 0:
            qp.save()
            qp.setPen(QtGui.QPen(
                QtGui.QBrush(QtGui.QColor(0, 255, 0)), 1.0))
            qp.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 0)))
            for control_point in spline_control_points:
                qp.drawEllipse(control_point, 2.0, 2.0)
            qp.restore()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

