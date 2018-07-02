#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yuanning Zhang (yuanningzhang@deepmotion.ai)
# Copyright (c) 2018-present, DeepMotion


import os
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
import scipy as sp
from timeit import default_timer as timer
import b_spline_model_3d
import sort_3d_data
import matplotlib.pyplot as plt
import math


class Lane3D():

    def __init__(self):
        pass

    def fit(self, data, displayed_points=100):

        # 1: Set a ceiling for number of data points used
        ceil = 100
        if len(data) > ceil:
            np.random.seed(0)
            data = np.random.permutation(data)[:ceil]

        # 2: Use RANSAC to eliminate outliers
        pca = PCA(n_components=1)
        pca.fit(data)
        if not (len(data) < 5 or pca.explained_variance_ratio_[0] > 0.999):
            data = self.RANSAC_Mixed_Interpolation(data, niter=100, threshold=self.span(data)/30.0)
        plt.plot(np.transpose(data)[0], np.transpose(data)[1], "b^")  # uncomment this line to see result of RANSAC

        # 3: Select some data points in a geometric order
        s3dd = sort_3d_data.Sort3DData(data, step_size=(self.span(data) / 10.0),
                                       tolerance=max(len(data)/50, 3), growthfactor=1.2)
        data = s3dd.getsorted()
        plt.plot(np.transpose(data)[0], np.transpose(data)[1], "ro")  # uncomment this line to see result of PCA

        # 4: Use linear model if data points behave linearly
        pca = PCA(n_components=1)
        newdata = pca.fit_transform(data)
        if len(data) < 5 or pca.explained_variance_ratio_[0] > 0.9995:
            newdata = pca.inverse_transform(newdata)
            ts = np.linspace(0, 1, displayed_points).reshape(displayed_points, 1)
            curve = np.dot(ts, [newdata[0]]) + np.dot(1 - ts,  [newdata[-1]])
            return curve

        # 5: Otherwise, use B-spline model with L-BFGS method
        n = len(data)
        numctrl = (n - 2) / 3
        ctrl = np.array([data[0], data[0], data[0]])
        for i in range(numctrl):
            ctrl = np.append(ctrl, data[(i + 1) * n / (numctrl + 1)])
        ctrl = np.append(ctrl, data[n - 1])
        ctrl = np.append(ctrl, data[n - 1])
        ctrl = np.append(ctrl, data[n - 1])
        ctrl = ctrl.reshape(len(ctrl) / 3, 3)
        model = b_spline_model_3d.BSplineModel3D(data, ctrl)
        curve = model.l_bfgs_fitting(displayed_points=displayed_points, maxf=2000)
        return curve

    def span(self, arr3d):
        arr3d = np.array(arr3d)
        temp = np.transpose(arr3d)
        xs, ys, zs = temp[0], temp[1], temp[2]
        dx = np.max(xs) - np.min(xs)
        dy = np.max(ys) - np.min(ys)
        dz = np.max(zs) - np.min(zs)
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def RANSAC_Mixed_Interpolation(self, data, niter=150, threshold=10.0):
        trans = np.transpose(data)
        dataxs, datays, datazs = trans[0], trans[1], trans[2]
        if len(dataxs) < 4:
            return data
        np.random.seed(0)
        max_gain = 0
        n = len(dataxs)
        best_dataxs = None
        best_datays = None
        best_datazs = None
        num = 0
        i = 0
        while True:
            newdataxs = []
            newdatays = []
            newdatazs = []
            if np.random.randint(2) == 0:  # use linear model
                rand1 = np.random.randint(n)
                rand2 = np.random.randint(n)
                while True:
                    if rand1 == rand2:
                        rand2 = np.random.randint(n)
                    else:
                        break
                polyx = sp.interpolate.lagrange(range(2), [dataxs[rand1], dataxs[rand2]])
                polyy = sp.interpolate.lagrange(range(2), [datays[rand1], datays[rand2]])
                polyz = sp.interpolate.lagrange(range(2), [datazs[rand1], datazs[rand2]])
                gain = 0
                M = 15
                j = 0
                l = len(dataxs)
                while True:
                    p3 = np.array([dataxs[j], datays[j], datazs[j]])
                    A = np.linspace(-1, 2, 3 * M + 1)
                    B = np.transpose(np.array([polyx(A), polyy(A), polyz(A)]))
                    min_point2curve = min(min(distance.cdist([p3], B)))
                    if min_point2curve <= threshold:
                        newdataxs.append(p3[0])
                        newdatays.append(p3[1])
                        newdatazs.append(p3[2])
                        gain = gain + 1
                    j += 1
                    if j == l:
                        break
                newdataxs = np.array(newdataxs)
                newdatays = np.array(newdatays)
                newdatazs = np.array(newdatazs)
                if gain > max_gain:
                    if gain - max_gain > n / 50:
                        num = 0
                    max_gain = gain
                    best_dataxs = newdataxs
                    best_datays = newdatays
                    best_datazs = newdatazs
                if gain == n:
                    break
                num = num + 1
            else:  # use quadratic model
                rand1 = np.random.randint(n)
                rand2 = np.random.randint(n)
                rand3 = np.random.randint(n)
                while True:
                    if rand1 == rand2:
                        rand2 = np.random.randint(n)
                    else:
                        break
                while True:
                    if rand3 == rand1 or rand3 == rand2:
                        rand3 = np.random.randint(n)
                    else:
                        break
                polyx = sp.interpolate.lagrange(range(3), [dataxs[rand1], dataxs[rand2], dataxs[rand3]])
                polyy = sp.interpolate.lagrange(range(3), [datays[rand1], datays[rand2], datays[rand3]])
                polyz = sp.interpolate.lagrange(range(3), [datazs[rand1], datazs[rand2], datazs[rand3]])
                gain = 0
                M = 10
                j = 0
                l = len(dataxs)
                while True:
                    p3 = np.array([dataxs[j], datays[j], datazs[j]])
                    A = np.linspace(-1, 3, 4 * M + 1)
                    B = np.transpose(np.array([polyx(A), polyy(A), polyz(A)]))
                    min_point2curve = min(min(distance.cdist([p3], B)))
                    if min_point2curve <= threshold * 0.75:
                        newdataxs.append(p3[0])
                        newdatays.append(p3[1])
                        newdatazs.append(p3[2])
                        gain = gain + 1
                    j += 1
                    if j == l:
                        break
                newdataxs = np.array(newdataxs)
                newdatays = np.array(newdatays)
                newdatazs = np.array(newdatazs)
                if gain > max_gain:
                    if gain - max_gain > 0:
                        num = 0
                    max_gain = gain
                    best_dataxs = newdataxs
                    best_datays = newdatays
                    best_datazs = newdatazs
                if gain == n:
                    break
                num = num + 1
                if num >= 75:
                    return np.transpose(np.array([best_dataxs, best_datays, best_datazs]))
            i += 1
            if i == niter or max_gain > 5 * n / 6:
                break
        return np.transpose(np.array([best_dataxs, best_datays, best_datazs]))


def read_off(file_path):
    f = open(file_path, 'r')
    f.readline()
    n_verts, n_faces, n_edges = tuple([int(s) for s in f.readline().strip().split(' ')])
    verts = np.array([])
    for i_vert in range(n_verts):
        verts = np.append(verts, np.array([float(s) for s in f.readline().strip().split(' ')]))
    verts = verts.reshape(n_verts, 3)
    f.close()
    return verts


def write_off(verts, file_path):
    n_verts = len(verts)
    f = open(file_path, 'w')
    f.write("OFF\n")
    f.write("{} 0 0\n".format(n_verts))
    for v in verts:
        f.write("{} {} {}\n".format(v[0], v[1], v[2]))
    f.close()


def set_color(in_file_path, r, g, b, a, out_file_path):
    fin = open(in_file_path, 'r')
    fout = open(out_file_path, 'w')
    fin.readline()
    fout.write("COFF\n")
    n_verts, n_faces, n_edges = tuple([int(s) for s in fin.readline().strip().split(' ')])
    fout.write("{} 0 0\n".format(n_verts))
    for i_vert in range(n_verts):
        v = np.array([float(s) for s in fin.readline().strip().split(' ')])[0:3]
        v = np.append(v, [r, g, b, a])
        fout.write("{} {} {} {} {} {} {}\n".format(v[0], v[1], v[2], v[3], v[4], v[5], v[6]))
    fin.close()
    fout.close()


if __name__ == '__main__':
    start = timer()

    pyplot_visualization = False
    data_path = "/home/yuanning/DeepMotion/pointclouds"
    files = sorted(os.listdir(data_path))
    n = len(files)

    for i in range(n):
        lane_mask = read_off(os.path.join(data_path, files[i]))

        lane3d = Lane3D()
        curve = lane3d.fit(lane_mask, displayed_points=1000)

        # pyplot_visualization = True  # uncomment this to see pyplot visualization instead of writing results to files
        if pyplot_visualization:
            plt.plot(np.transpose(lane_mask)[0], np.transpose(lane_mask)[1], "b,")  # uncomment to see original data
            plt.plot(np.transpose(curve)[0], np.transpose(curve)[1], 'r')
            plt.show()
        else:
            curve_path = "/home/yuanning/DeepMotion/curves/curve_{}".format(files[i])
            write_off(curve, curve_path)
            colorized_curve_path = "/home/yuanning/DeepMotion/colorized_curves/colorized_curve_{}".format(files[i])
            set_color(curve_path, 1.0, 0.0, 0.0, 1.0, colorized_curve_path)
        print(files[i] + " has been fitted.")
    plt.show()

    if not pyplot_visualization:
        print("All fitting curves have been written to files.")

    end = timer()
    print("Total time used: {}".format(end - start))
