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


class Node(object):

    def __init__(self, parent=None):
        self.points = []
        self.parent = parent
        self.leaves = [0, 0, 0, 0]
        self.level = 0
        self.index_list = []
        if parent is not None:
            self.level = self.parent.level + 1


class QuadTree(object):

    def __init__(self, height, width, max_level=5):
        self.height = height
        self.width = width
        self.max_level = max_level
        self.num_leaf = 0
        self.root = Node()
        self.all_leaves = {}

    def store(self, p):
        x = p[0]
        y = p[1]
        h = self.height
        w = self.width
        node = self.root
        index_list = []
        for i in range(self.max_level):
            index = 0
            if x % w >= w / 2.0:
                index = index + 1
            if y % h >= h / 2.0:
                index = index + 2
            if node.leaves[index] is 0:
                node.leaves[index] = Node(parent=node)
                if self.max_level is node.level + 1:
                    self.num_leaf = self.num_leaf + 1
            node = node.leaves[index]
            index_list.append(index)
            node.index_list = index_list
            x = x % w
            y = y % h
            w = w / 2.0
            h = h / 2.0
        node.points.append([x, y])
        return index_list

    def getindex(self, x, y):
        index_list = []
        h = self.height
        w = self.width
        for i in range(self.max_level):
            index = 0
            if w * h > 0:
                if x % w >= w / 2.0:
                    index = index + 1
                if y % h >= h / 2.0:
                    index = index + 2
            index_list.append(index)
            x = x % w
            y = y % h
            w = w / 2.0
            h = h / 2.0
        return index_list

    def get(self, index_list):
        node = self.root
        for index in index_list:
            node = node.leaves[index]
        return node.points

    def getneighbors(self, p):
        x = p[0]
        y = p[1]
        h = self.height / float(2 ** self.max_level)
        w = self.width / float(2 ** self.max_level)
        neighbor_set = []
        if x - w >= 0 and y - h >= 0:
            neighbor_set.append(self.getindex(x - w, y - h))
        if y - h >= 0:
            neighbor_set.append(self.getindex(x, y - h))
        if x + w < self.width and y - h >= 0:
            # print(self.getindex(x + w, y - h))
            neighbor_set.append(self.getindex(x + w, y - h))
            # print(self.width)
            # print(x + w)
        if x + w < self.width:
            neighbor_set.append(self.getindex(x + w, y))
        if x + w < self.width and y + h < self.height:
            neighbor_set.append(self.getindex(x + w, y + h))
        if y + h < self.height:
            neighbor_set.append(self.getindex(x, y + h))
            # print(self.getindex(x, y))
            # print(self.getindex(x, y + h))
            # print(self.height)
            # print(y)
            # print(y + h)
        if x - w >= 0 and y + h < self.height:
            neighbor_set.append(self.getindex(x - w, y + h))
        if x - w >= 0:
            neighbor_set.append(self.getindex(x - w, y))
        if x - w >= 0 and y - h >= 0:
            neighbor_set.append(self.getindex(x - w, y - h))
        return neighbor_set

    def getcenter(self, index_list):
        x = 0
        y = 0
        h = self.height / 2.0
        w = self.width / 2.0
        for index in index_list:
            if index % 2 is 1:
                x = x + w
            if index / 2 is 1:
                y = y + h
            h = h / 2.0
            w = w / 2.0
        x = x + w
        y = y + h
        return [x, y]


if __name__ == '__main__':
    start = time.time()
    data_path = '/home/yuanning/DeepMotion/lane/data'
    files = sorted(os.listdir(data_path))
    lane_mask = np.load(os.path.join(data_path, files[4]))
    # for fname in files:
    #     data = np.load(os.path.join(data_path, fname))
    #     cv2.imshow("Lane", LaneMath.colorise(data, True))
    #     cv2.waitKey(1)
    # white_background = 255 - np.zeros(shape=(64, 64, 3))
    # plt.imshow(white_background)
    lane_mask = lane_mask[300:, 500:1400]
    img = LaneMath.colorise(lane_mask)
    plt.imshow(img)
    lane_ids = np.unique(lane_mask)
    lane_ids = lane_ids[1:]
    lanes = {}
    for id in lane_ids:
        lanes[id] = []
    for i in range(lane_mask.shape[0]):
        for j in range(lane_mask.shape[1]):
            if lane_mask[i, j] in lane_ids:
                lanes[lane_mask[i, j]].append([j, i])
    points = []
    for id in lane_ids:
        dataxs, datays, use_PCA = LaneMath.get_data_sample(lane_mask, id, lanes)
        points.extend([[dataxs[k], datays[k]] for k in range(len(dataxs))])
    qt = QuadTree(img.shape[0], img.shape[1])
    # points = [[4, 8], [15, 16], [22, 42], [20, 30], [10, 10], [17, 22], [21, 38], [22, 48]]
    # qt = QuadTree(64, 64)
    # points = points[4:5]
    # print(points[0])
    for p in points:
        plt.plot(p[0], p[1], "ko")
        qt.store(p)
        # print(qt.store(p))
    for p in points:
        # center = qt.getcenter(qt.getindex(p[0], p[1]))
        # plt.plot(center[0], center[1], "b^")
        centers = []
        for index_list in qt.getneighbors(p):
            centers.append(qt.getcenter(index_list))
        centerxs, centerys = [center[0] for center in centers], [center[1] for center in centers]
        plt.plot(centerxs, centerys, "r")
    for p in points:
        center = qt.getcenter(qt.getindex(p[0], p[1]))
        # print(center)
        plt.plot(center[0], center[1], "b^")
    # print()
    # for p in points:
    #     print(qt.getindex(p[0], p[1]))
    # print(qt.get([2, 1, 2, 1]))
    plt.show()
    end = time.time()
    print("Time used: " + str(end - start))
