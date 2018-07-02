#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yuanning Zhang (yuanningzhang@deepmotion.ai)
# Copyright (c) 2018-present, DeepMotion


import numpy as np
from scipy.spatial import kdtree
from sklearn.decomposition import PCA


# This class implements an algorithm that finds a (geometrically) sorted sample of a given 3D point cloud.
class Sort3DData(object):

    def __init__(self, points, step_size=30, tolerance=20, growthfactor=1.1):
        np.random.seed(0)
        self.step_size = step_size
        self.unsorted = np.array(points)
        self.n = len(points)
        self.sortedxs = []  # these will store the sorted data points returned by this algorithm
        self.sortedys = []
        self.sortedzs = []
        self.kdtree = kdtree.KDTree(points)
        self.tolerance = tolerance
        self.growthfactor = growthfactor
        initpoint = self.unsorted[self.n / 2]  # select an initial data point
        i = self.step_size
        while True:  # enlarge the searching area to get enough data points
            neighbors = self.kdtree.query_ball_point(initpoint, i)
            if len(neighbors) >= self.tolerance or i > self.step_size * 2:
                break
            i = i * self.growthfactor
        pca = PCA(n_components=1)
        pca.fit(self.unsorted[neighbors])
        initdirection1 = pca.components_[0]  # PCA helps to find the local direction of our initial point
        initdirection2 = -initdirection1

        # Step 1: search along the first direction and store sorted samples
        p = initpoint
        d = initdirection1
        oldp = p
        while d is not None:
            p, d = self.searchnext(p, d)
            if p is None or (oldp[0] == p[0] and oldp[1] == p[1] and oldp[2] == p[2]):
                break
            oldp = p
            if d is not None:
                arr1 = np.array([p[0]])
                arr2 = np.array([p[1]])
                arr3 = np.array([p[2]])
                if len(self.sortedxs) == 0:
                    self.sortedxs = arr1
                    self.sortedys = arr2
                    self.sortedzs = arr3
                else:
                    self.sortedxs = np.concatenate((arr1, self.sortedxs))
                    self.sortedys = np.concatenate((arr2, self.sortedys))
                    self.sortedzs = np.concatenate((arr3, self.sortedzs))
        arr1 = np.array([p[0]])
        arr2 = np.array([p[1]])
        arr3 = np.array([p[2]])
        if len(self.sortedxs) == 0:
            self.sortedxs = arr1
            self.sortedys = arr2
            self.sortedzs = arr3
        else:
            self.sortedxs = np.concatenate((arr1, self.sortedxs))
            self.sortedys = np.concatenate((arr2, self.sortedys))
            self.sortedzs = np.concatenate((arr3, self.sortedzs))

        # Step 2: append the initial point to sorted samples
        self.sortedxs = np.append(self.sortedxs, initpoint[0])
        self.sortedys = np.append(self.sortedys, initpoint[1])
        self.sortedzs = np.append(self.sortedzs, initpoint[2])

        # Step 3: search along the second direction and append new sorted samples to previous samples
        p = initpoint
        d = initdirection2
        oldp = p
        while d is not None:  # search along the second direction
            p, d = self.searchnext(p, d)
            if d is None or (oldp[0] == p[0] and oldp[1] == p[1] and oldp[2] == p[2]):
                break
            oldp = p
            if d is not None:
                self.sortedxs = np.append(self.sortedxs, p[0])
                self.sortedys = np.append(self.sortedys, p[1])
                self.sortedzs = np.append(self.sortedzs, p[2])

        self.sortedxs = np.append(self.sortedxs, p[0])
        self.sortedys = np.append(self.sortedys, p[1])
        self.sortedzs = np.append(self.sortedzs, p[2])

        self.sorted = np.transpose(np.array([self.sortedxs, self.sortedys, self.sortedzs]))


    def searchnext(self, p, direction):
        i = self.step_size
        while True:
            neighbors = self.kdtree.query_ball_point(p + direction * i, i)
            if len(neighbors) >= self.tolerance or i > self.step_size * 2:
                break
            i = i * self.growthfactor
        if len(neighbors) < self.tolerance:
            dists, ps = self.kdtree.query([p + direction * i])
            newp = self.unsorted[ps[0]]
            return newp, None
        dists, ps = self.kdtree.query([p + direction * i])
        newp = self.unsorted[ps[0]]
        pca = PCA(n_components=1)
        pca.fit(self.unsorted[neighbors])
        newdirection = pca.components_[0]
        if np.dot(direction, newdirection) < 0:
            newdirection = -newdirection
        return newp, newdirection

    def getsorted(self):
        return self.sorted
