#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yuanning Zhang (yuanningzhang@deepmotion.ai)
# Copyright (c) 2018-present, DeepMotion


import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial import kdtree
from sklearn.decomposition import PCA


class Sort3DData(object):

    def __init__(self, points, step_size=30, tolerance=50):
        self.step_size = step_size
        np.random.seed(0)
        self.unsorted = np.array(points)
        self.n = len(points)
        self.sortedxs = []
        self.sortedys = []
        self.sortedzs = []
        self.kdtree = kdtree.KDTree(points)
        self.tolerance = tolerance
        initpoint = self.unsorted[self.n / 2]
        i = self.step_size
        while True:
            i = i * 1.1
            neighbors = self.kdtree.query_ball_point(initpoint, i)
            if len(neighbors) >= self.tolerance or i > self.step_size * 10:
                break
        pca = PCA(n_components=1)
        pca.fit(self.unsorted[neighbors])
        initdirection1 = pca.components_[0]
        initdirection2 = -initdirection1
        p = initpoint
        d = initdirection1
        oldp = p
        while p is not None:
            p, d = self.searchnext(p, d)
            if p is None or (oldp[0] == p[0] and oldp[1] == p[1] and oldp[2] == p[2]):
                break
            oldp = p
            if p is not None:
                arr1 = np.zeros(1)
                arr2 = np.zeros(1)
                arr3 = np.zeros(1)
                arr1[0] = p[0]
                arr2[0] = p[1]
                arr3[0] = p[2]
                if len(self.sortedxs) == 0:
                    self.sortedxs = arr1
                    self.sortedys = arr2
                    self.sortedzs = arr3
                else:
                    self.sortedxs = np.concatenate((arr1, self.sortedxs))
                    self.sortedys = np.concatenate((arr2, self.sortedys))
                    self.sortedzs = np.concatenate((arr3, self.sortedzs))
        self.sortedxs = np.append(self.sortedxs, initpoint[0])
        self.sortedys = np.append(self.sortedys, initpoint[1])
        self.sortedzs = np.append(self.sortedzs, initpoint[2])
        p = initpoint
        d = initdirection2
        oldp = p
        while p is not None:
            p, d = self.searchnext(p, d)
            if p is None or (oldp[0] == p[0] and oldp[1] == p[1] and oldp[2] == p[2]):
                break
            oldp = p
            if p is not None:
                self.sortedxs = np.append(self.sortedxs, p[0])
                self.sortedys = np.append(self.sortedys, p[1])
                self.sortedzs = np.append(self.sortedzs, p[2])
        self.sorted = np.transpose(np.array([self.sortedxs, self.sortedys, self.sortedzs]))


    def searchnext(self, p, direction):
        i = self.step_size
        while True:
            i = i * 1.1
            neighbors = self.kdtree.query_ball_point(p + direction * i * 1.5, i)
            if len(neighbors) == 0:
                return None, None
            if len(neighbors) >= self.tolerance or i > self.step_size * 10:
                break
        if len(neighbors) < self.tolerance:
            return None, None
        dists, ps = self.kdtree.query([p + direction * i * 1.2])
        newp = self.unsorted[ps[0]]
        pca = PCA(n_components=1)
        pca.fit(self.unsorted[neighbors])
        newdirection = pca.components_[0]
        if np.dot(direction, newdirection) < 0:
            newdirection = -newdirection
        return newp, newdirection

    def getsorted(self):
        return self.sorted


# if __name__ == '__main__':
#     start = time.time()
#     plt.imshow(255 - np.zeros((64, 64, 3)))
#     data_points = [[30, 35], [40, 40], [20, 20], [50, 42], [20, 25], [23, 29], [27, 33], [43, 41],
#                    [30, 12], [36, 10], [47, 41], [25, 15], [35, 38], [24, 16], [22, 18], [20, 22],
#                    [33, 37], [32, 11], [27, 13]]
#     sd = SortData(data_points, step_size=2)
#     data = sd.getsorted()
#     print(data)
#     plt.plot([p[0] for p in data_points], [p[1] for p in data_points], 'b^')
#     plt.plot([p[0] for p in data], [p[1] for p in data], 'r^--')
#     plt.show()
#     end = time.time()
#     print("Time used: " + str(end - start))
