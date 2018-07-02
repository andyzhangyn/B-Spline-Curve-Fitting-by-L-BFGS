#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yuanning Zhang (yuanningzhang@deepmotion.ai)
# Copyright (c) 2018-present, DeepMotion


import numpy as np
import scipy as sp
import cubic_b_spline_3d
import scipy.optimize


class BSplineModel3D(object):

    def __init__(self, data, pts, img=None, id=None):
        self.data = data
        self.pts = pts
        self.N = data.shape[0]
        self.n = pts.shape[0]
        self.cbs = cubic_b_spline_3d.CubicBSpline3D(self.pts)
        self.T = np.zeros(self.N)
        k = (self.n - 3.0) / self.N
        for i in range(self.N):
            self.T[i] = 3 + i * k
        self.img = img
        self.id = id

    def l_bfgs_fitting(self, displayed_points=100, maxf=1000):
        A = np.transpose(self.pts)
        xs = A[0]
        ys = A[1]
        zs = A[2]
        x = np.concatenate((xs, ys, zs, self.T))
        x, min_val, info = sp.optimize.fmin_l_bfgs_b(self.loss, x, approx_grad=True, m=8, maxfun=maxf)
        for k in range(self.n):
            self.pts[k] = [x[k], x[k+self.n], x[k+2*self.n]]
        self.cbs = cubic_b_spline_3d.CubicBSpline3D(self.pts)
        curve = np.array([])
        ts = np.linspace(3, k, displayed_points)
        for t in ts:
            curve = np.append(curve, self.cbs.eval(t))
        return curve.reshape(displayed_points, 3)

    def loss(self, v):
        newpts = np.zeros((self.n, 3))
        for k in range(self.n):
            newpts[k] = [v[k], v[k + self.n], v[k + 2 * self.n]]
        newcbs = cubic_b_spline_3d.CubicBSpline3D(newpts)
        newT = v[3 * self.n:]
        sum = 0
        for i in range(self.N):
            p1 = newcbs.eval(newT[i])
            p2 = self.data[i]
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            dz = p1[2] - p2[2]
            sum = sum + dx * dx + dy * dy + dz * dz
        sum = sum * self.N
        return sum
