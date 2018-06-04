#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yuanning Zhang (yuanningzhang@deepmotion.ai)
# Copyright (c) 2018-present, DeepMotion

import os
import cv2
# from lane_math import LaneMath
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.spatial import kdtree
from sklearn.decomposition import PCA


class SortData(object):

    def __init__(self, points, step_size=30, tolerance=50):
        self.step_size = step_size
        np.random.seed(0)
        self.unsorted = np.array(points)
        self.n = len(points)
        self.sortedxs = []
        self.sortedys = []
        self.kdtree = kdtree.KDTree(points)
        self.tolerance = tolerance
        # initpoint = self.unsorted[np.random.randint(self.n)]
        initpoint = self.unsorted[self.n / 2]
        # print(initpoint)

        i = self.step_size
        while True:
            # print(i)
            i = i * 2
            neighbors = self.kdtree.query_ball_point(initpoint, i)
            # print(neighbors)
            # print(i)
            if len(neighbors) >= self.tolerance or i > self.step_size * 3:
                break
            # print(step)
            # step = step * 2
            # print(step)
            # i = i + 2

        # neighbors = self.kdtree.query_ball_point(initpoint, self.step_size)
        # print(neighbors)
        pca = PCA(n_components=2)
        # pts = self.unsorted[neighbors]
        # plt.plot([p[0] for p in pts], [p[1] for p in pts], 'ro')
        pca.fit(self.unsorted[neighbors])
        initdirection1 = pca.components_[0]
        initdirection2 = -initdirection1
        # temp = pca.transform(self.unsorted[neighbors])
        p = initpoint
        # print(p)
        d = initdirection1
        oldp = p
        while p is not None:
            p, d = self.searchnext(p, d)
            if p is None or (oldp[0] == p[0] and oldp[1] == p[1]):
                break
            oldp = p
            # print(p)
            if p is not None:
                # print(p)
                # print(self.sorted)
                arr1 = np.zeros(1)
                arr2 = np.zeros(1)
                arr1[0] = p[0]
                arr2[0] = p[1]
                if len(self.sortedxs) == 0:
                    self.sortedxs = arr1
                    self.sortedys = arr2
                    # print("hello")
                else:
                    self.sortedxs = np.concatenate((arr1, self.sortedxs))
                    self.sortedys = np.concatenate((arr2, self.sortedys))
            # print(self.sortedxs)
            # print(self.sortedys)
        self.sortedxs = np.append(self.sortedxs, initpoint[0])
        self.sortedys = np.append(self.sortedys, initpoint[1])
        p = initpoint
        d = initdirection2
        oldp = p
        while p is not None:
            p, d = self.searchnext(p, d)
            if p is None or (oldp[0] == p[0] and oldp[1] == p[1]):
                break
            oldp = p
            if p is not None:
                self.sortedxs = np.append(self.sortedxs, p[0])
                self.sortedys = np.append(self.sortedys, p[1])
        self.sorted = [[self.sortedxs[k], self.sortedys[k]] for k in range(len(self.sortedxs))]
        # print(self.sorted)


    def searchnext(self, p, direction):
        unit_direction = direction
        # unit_direction = direction / math.sqrt(direction[0] ** 2 + direction[1] ** 2)
        # print(unit_direction)
        # p = p + unit_direction * self.step_size

        # neighbors = self.kdtree.query_ball_point(p, self.step_size)
        # if len(neighbors) < 2:
        #     return None, None

        # step = self.step_size
        neighbors = []
        i = self.step_size
        while True:
            # print(i)
            i = i * 1.1
            neighbors = self.kdtree.query_ball_point(p + unit_direction * i * 1.5, i)
            # print(neighbors)
            # print(i)
            if len(neighbors) == 0:
                # print("Hello")
                return None, None

            if len(neighbors) >= self.tolerance or i > self.step_size * 3:
                break
            # print(step)
            # step = step * 2
            # print(step)
            # i = i + 2
        # print(neighbors)
        if len(neighbors) < self.tolerance:
            return None, None
        dists, ps = self.kdtree.query([p + unit_direction * i * 1.2])
        newp = self.unsorted[ps[0]]
        pca = PCA(n_components=1)
        pca.fit(self.unsorted[neighbors])
        newdirection = pca.components_[0]
        if np.dot(direction, newdirection) < 0:
            newdirection = -newdirection
        # print(newp)
        return newp, newdirection

    def getsorted(self):
        return self.sorted


if __name__ == '__main__':
    start = time.time()
    plt.imshow(255 - np.zeros((64, 64, 3)))
    data_points = [[30, 35], [40, 40], [20, 20], [50, 42], [20, 25], [23, 29], [27, 33], [43, 41],
                   [30, 12], [36, 10], [47, 41], [25, 15], [35, 38], [24, 16], [22, 18], [20, 22],
                   [33, 37], [32, 11], [27, 13]]
    sd = SortData(data_points, step_size=2)
    data = sd.getsorted()
    print(data)
    plt.plot([p[0] for p in data_points], [p[1] for p in data_points], 'b^')
    plt.plot([p[0] for p in data], [p[1] for p in data], 'r^--')
    plt.show()
    end = time.time()
    print("Time used: " + str(end - start))