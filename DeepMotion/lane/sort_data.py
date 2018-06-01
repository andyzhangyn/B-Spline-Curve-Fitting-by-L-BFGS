#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yuanning Zhang (yuanningzhang@deepmotion.ai)
# Copyright (c) 2018-present, DeepMotion

import os
import cv2
from lane_math import LaneMath
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.spatial import kdtree
from sklearn.decomposition import PCA


class SortData(object):

    def __init__(self, points):
        np.random.seed(0)
        self.unsorted = np.array(points)
        self.n = len(points)
        self.sorted = np.empty((0, 2))
        self.kdtree = kdtree.KDTree(points)
        initpoint = self.unsorted[np.random.randint(self.n)]
        print(initpoint)
        neighbors = self.kdtree.query_ball_point(initpoint, 5)
        pca = PCA(n_components=1)
        pca.fit(self.unsorted[neighbors])
        initdirection1 = pca.components_[0]
        initdirection2 = -initdirection1
        p = initpoint
        d = initdirection1
        while p is not None:
            p, d = self.searchnext(p, d)
            if p is not None:
                self.sorted = np.concatenate(([p], self.sorted))
        np.append(self.sorted, initpoint)
        p = initpoint
        d = initdirection2
        while p is not None:
            p, d = self.searchnext(p, d)
            if p is not None:
                np.append(self.sorted, p)


    def searchnext(self, p, direction):
        unit_direction = direction / math.sqrt(direction[0] ** 2 + direction[1] ** 2)
        p = p + unit_direction * 5
        neighbors = self.kdtree.query_ball_point(p, 5)
        if len(neighbors) is 0:
            return None, None
        newp = self.kdtree.query(p)
        pca = PCA(n_components=1)
        pca.fit(self.unsorted[neighbors])
        newdirection = pca.components_[0]
        return newp, newdirection

    def getsorted(self):
        return self.sorted


if __name__ == '__main__':
    start = time.time()
    plt.imshow(255 - np.zeros((64, 64, 3)))
    data_points = [[30, 35], [40, 40], [20, 20], [50, 42], [20, 25], [23, 29], [27, 33], [43, 41],
                   [30, 12], [36, 10], [47, 41], [25, 15], [35, 38], [24, 16], [22, 18], [20, 22],
                   [33, 37], [32, 11], [27, 13]]
    sd = SortData(data_points)
    data = sd.getsorted()
    plt.plot([p[0] for p in data_points], [p[1] for p in data_points], 'b^')
    plt.plot([p[0] for p in data], [p[1] for p in data], 'rs')
    plt.show()
    end = time.time()
    print("Time used: " + str(end - start))