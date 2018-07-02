#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yuanning Zhang (yuanningzhang@deepmotion.ai)
# Copyright (c) 2018-present, DeepMotion


import numpy as np


class CubicBSpline3D(object):

    def __init__(self, pts):
        self.n = pts.shape[0]
        self.pts = pts

        def eval(t):
            xyz = np.zeros(3)
            for k in range(self.n):
                b = self.cubic(k, t)
                xyz = xyz + self.pts[k] * b
            return xyz

        self.eval = eval

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
