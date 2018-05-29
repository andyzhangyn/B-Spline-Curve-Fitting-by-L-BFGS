#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yuanning Zhang (yuanningzhang@deepmotion.ai)
# Copyright (c) 2018-present, DeepMotion


import numpy as np


class CubicBSpline(object):

    def __init__(self, pts):
        self.n = pts.shape[0]
        self.pts = pts

        def eval(t):
            xy = [0, 0]
            for k in range(self.n):
                point = self.pts[k]
                b = self.cubic(k, t)
                xy[0] = xy[0] + point[0] * b
                xy[1] = xy[1] + point[1] * b
            return np.array(xy)

        self.eval = eval
        # ts = np.linspace(3, self.n, self.M)
        # for i in range(self.M):
        #     self.points[ts[i]] = self.eval(ts[i])

    def generate(self, newts):
        newpts = np.zeros((newts.shape[0], 2))
        for i in range(newts.shape[0]):
            newpts[i, 0], newpts[i, 1] = self.eval(newts[i])
        return newpts

    # @staticmethod
    # def quadratic(k, u):  # quadratic uniform basis
    #     u = u - k
    #     if u < 0:
    #         return 0
    #     if u < 1:
    #         return 0.5 * u * u
    #     if u < 2:
    #         return u * (3 - u) - 1.5
    #     if u < 3:
    #         return 0.5 * (3 - u) * (3 - u)
    #     return 0

    # def cubic(self, k, u):  # cubic uniform basis
    #     return (self.quadratic(k, u) * (u - k) + self.quadratic(k+1, u) * (k - u + 4)) / 3

    @staticmethod
    def cubic(k, u):
        u = u - k
        if u < 0:
            return 0
        if u < 1:
            return u * u * u / 6.0
        if u < 2:
            return 2.0 / 3.0 - ((u - 4.0) * u + 4.0) * u / 2.0
        if u < 3:
            return ((u - 8.0) * u + 20.0) * u / 2.0 - 22.0 / 3.0
        if u < 4:
            w = 4.0 - u
            return w * w * w / 6.0
        return 0
