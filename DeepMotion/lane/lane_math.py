#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yuanning Zhang (yuanningzhang@deepmotion.ai)
# Copyright (c) 2018-present, DeepMotion


import os
import numpy as np
import matplotlib.pyplot as plt
import b_spline_model
import time
from sklearn.decomposition import PCA
import math
import scipy as sp
import sort_data
from scipy.spatial import distance
from timeit import default_timer as timer


class LaneMath(object):

    def __init__(self):
        self.time = 0.0
        pass

    def show(self, data_path, axis=False):
        files = sorted(os.listdir(data_path))
        # k = 0
        # for file in files:
        #     lane_mask = np.load(os.path.join(data_path, file))
        #     self.show_graph(lane_mask, axis)
        #     print("File " + str(k) + " completed")
        #     k = k + 1
        # start = time.time()
        # for k in range(400, 410):
        #     lane_mask = np.load(os.path.join(data_path, files[k]))
        #     self.show_graph(lane_mask, axis)
        # end = time.time()
        # print(end - start)
        lane_mask = np.load(os.path.join(data_path, files[4]))
        self.show_graph(lane_mask, axis)
        print("Time used for subprogram: " + str(self.time))
        return self.time

    def show_graph(self, lane_mask, axis=False):
        lane_mask = lane_mask[300:, 500:1400]
        img = self.colorise(lane_mask)
        plt.imshow(img)
        if not axis:
            plt.axis('off')
        lane_ids = np.unique(lane_mask)
        lane_ids = lane_ids[1:]
        lanes = {}
        lanes_trans = {}
        A = np.transpose(lane_mask)
        B = lane_mask
        for id in lane_ids:
            lanes_trans[id] = np.transpose(np.nonzero(A == id))
            C = np.array(np.nonzero(B == id))
            C[[0, 1]] = C[[1, 0]]
            lanes[id] = np.transpose(C)
        for id in lane_ids:
            a = timer()
            if len(lanes[id]) < 800:
                continue
            data = lanes[id]
            data_2 = lanes_trans[id]
            pca = PCA(n_components=1)
            newdata = pca.fit_transform(data)
            extreme = pca.inverse_transform(min(newdata))
            if abs(pca.components_[0][0]) > abs(pca.components_[0][1]):
                data = lanes_trans[id]
                data_2 = lanes[id]
            d1 = (extreme[0] - pca.mean_[0]) ** 2 + (extreme[1] - pca.mean_[1]) ** 2
            d2 = (data[0][0] - pca.mean_[0]) ** 2 + (data[0][1] - pca.mean_[1]) ** 2
            d3 = (data[-1][0] - pca.mean_[0]) ** 2 + (data[-1][1] - pca.mean_[1]) ** 2
            d4 = (data_2[0][0] - pca.mean_[0]) ** 2 + (data_2[0][1] - pca.mean_[1]) ** 2
            d5 = (data_2[-1][0] - pca.mean_[0]) ** 2 + (data_2[-1][1] - pca.mean_[1]) ** 2
            if pca.explained_variance_ratio_[0] > 0.996 or \
                    ((d1 > d4 or d1 > d5) and (d1 > d2 or d1 > d3)) or len(data) < 1000:
                mindata = min(newdata)
                maxdata = max(newdata)
                newdata = pca.inverse_transform([mindata, maxdata])
                newdataxs = np.array([newdata[0][0], newdata[-1][0]])
                newdatays = np.array([newdata[0][1], newdata[-1][1]])
                plt.plot(newdataxs, newdatays, "r")
                continue
            A = np.transpose(data)
            dataxs, datays = self.RANSAC_Quadratic_Interpolation(dataxs, datays, id=id)
            # plt.plot(dataxs, datays, 'b,')
            sd = sort_data.SortData([[dataxs[k], datays[k]] for k in range(len(dataxs))], step_size=30, tolerance=50)
            data = sd.getsorted()
            # plt.plot([p[0] for p in data], [p[1] for p in data], 'rs')
            A = np.transpose(data)
            dataxs, datays = A[0], A[1]
            # plt.imshow(255 - np.zeros((lane_mask.shape[0], lane_mask.shape[1], 3)))
            # plt.show()
            pca = PCA(n_components=1)
            newdata = pca.fit_transform(data)
            if len(data) < 5 or pca.explained_variance_ratio_[0] > 0.999:
                newdata = pca.inverse_transform(newdata)
                newdataxs = np.array([newdata[0][0], newdata[-1][0]])
                newdatays = np.array([newdata[0][1], newdata[-1][1]])
                plt.plot(newdataxs, newdatays, "r")
                # b = timer()
                # self.time = self.time + b - a
                continue
            n = len(dataxs)
            data = np.transpose(np.array([dataxs, datays]))
            numctrl = len(lanes[id]) / 800
            arr = [[dataxs[0], datays[0]], [dataxs[0], datays[0]], [dataxs[0], datays[0]]]
            for i in range(numctrl - 1):
                arr.append([dataxs[(i + 1) * n / numctrl], datays[(i + 1) * n / numctrl]])
                # plt.plot(dataxs[(i + 1) * n / numctrl], datays[(i + 1) * n / numctrl], 'ko')
            arr.append([dataxs[n - 1], datays[n - 1]])
            arr.append([dataxs[n - 1], datays[n - 1]])
            arr.append([dataxs[n - 1], datays[n - 1]])
            ctrlpts = np.array(arr, dtype=np.float)
            model = b_spline_model.BSplineModel(data, ctrlpts, img=img, id=id)
            model.l_bfgs_fitting()
            b = timer()
            self.time = self.time + b - a
        plt.show()
        # self.time = self.time + b - a

    def RANSAC_Quadratic_Interpolation(self, dataxs, datays, niter=100, threshold=12, d=1000, id=None):
        if len(dataxs) < 4:
            return dataxs, datays
        np.random.seed(0)
        max_gain = 0
        n = len(dataxs)
        best_polyx = None
        best_polyy = None
        num = 0
        for i in range(niter):
            newdataxs = []
            newdatays = []
            K = 6
            if n >= K:
                rand1 = np.random.randint(0, n / K)
                rand2 = np.random.randint(n / K, (K - 1) * n / K)
                rand3 = np.random.randint((K - 1) * n / K, n)
            else:
                rand1 = 0
                rand2 = n / 2 - 1
                rand3 = n - 1

            # rand1 = np.random.randint(n)
            # rand2 = np.random.randint(n)
            # rand3 = np.random.randint(n)
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
            # var1 = min(rand1, rand2, rand3)
            # var3 = max(rand1, rand2, rand3)
            # var2 = rand1 + rand2 + rand3 - var1 - var3
            # rand1, rand2, rand3 = var1, var2, var3
            polyx = sp.interpolate.lagrange(range(3), [dataxs[rand1], dataxs[rand2], dataxs[rand3]])
            polyy = sp.interpolate.lagrange(range(3), [datays[rand1], datays[rand2], datays[rand3]])
            gain = 0
            M = 15
            for j in range(len(dataxs)):
                p3 = np.array([dataxs[j], datays[j]])
                A = np.linspace(-1, 3, 4 * M)
                B = np.transpose(np.array([polyx(A), polyy(A)]))
                min_point2curve = min(min(distance.cdist([p3], B)))
                if min_point2curve <= threshold:
                    newdataxs.append(p3[0])
                    newdatays.append(p3[1])
                    gain = gain + 1
            newdataxs = np.array(newdataxs)
            newdatays = np.array(newdatays)
            if len(newdataxs) >= 5 * n / 6 or num >= 30:
                return newdataxs, newdatays
            if gain > max_gain:
                if gain - max_gain > 30:
                    num = 0
                max_gain = gain
                best_dataxs = newdataxs
                best_datays = newdatays
                best_polyx = polyx
                best_polyy = polyy
            if gain == n:
                break
            num = num + 1
        print("Hello")
        # ts = np.linspace(-1, 3, 40)
        # if best_polyx is None or best_polyy is None:
        #     print("Lane " + str(id) + " RANSAC failed")
        # else:
        #     plt.plot(best_polyx(ts), best_polyy(ts), "k")
        return best_dataxs, best_datays


    def RANSAC_LeastSquare(self, dataxs, datays, niter=100, threshold=80, d=200):
        if len(dataxs) < 3:
            return dataxs, datays
        np.random.seed(0)
        best_dataxs = dataxs
        best_datays = datays
        min_loss = 1000000
        for i in range(niter):
            newdataxs = []
            newdatays = []
            rand1 = np.random.randint(0, len(dataxs) / 2)
            rand2 = np.random.randint(len(dataxs) / 2, len(dataxs))
            while True:
                if rand1 == rand2:
                    rand2 = np.random.randint(0, len(dataxs))
                else:
                    break
            p1 = np.array([dataxs[rand1], datays[rand1]])
            p2 = np.array([dataxs[rand2], datays[rand2]])
            v = p1 - p2
            loss = 0
            for j in range(len(dataxs)):
                p3 = np.array([dataxs[j], datays[j]])
                w = p3 - p2
                sq_dist = w[0] ** 2 + w[1] ** 2 - (v[0] * w[0] + v[1] * w[1]) ** 2 / (v[0] ** 2 + v[1] ** 2)
                if sq_dist < 0:
                    sq_dist = 0
                loss = loss + sq_dist
                dist = math.sqrt(sq_dist)
                if dist <= threshold:
                    newdataxs.append(p3[0])
                    newdatays.append(p3[1])
            newdataxs = np.array(newdataxs)
            newdatays = np.array(newdatays)
            if len(newdataxs) >= d:
                return newdataxs, newdatays
            if loss < min_loss:
                min_loss = loss
                best_dataxs = newdataxs
                best_datays = newdatays
        return best_dataxs, best_datays

    @staticmethod
    def colorise(lane_mask, black_background=False):
        np.random.seed(0)
        palette = np.random.randint(70, 255, size=(50, 3), dtype=np.uint8)
        lane_ids = np.unique(lane_mask)
        shape = lane_mask.shape
        img = np.zeros(shape=(shape[0], shape[1], 3), dtype=np.uint8)
        for i, lane_id in enumerate(lane_ids[1:]):
            img[lane_mask[:] == lane_id] = palette[i]
        if not black_background:
            img = 255 - img
        return img

    @staticmethod
    def get_data_sample(lane_mask, id, lanes):
        np.random.seed(0)
        dataxs, datays = [], []
        data_density = 100
        k = 0
        lane = lanes[id]
        while k < len(lane):
            dataxs.append(lane[k][0])
            datays.append(lane[k][1])
            k = k + data_density
        dataxs.append(lane[-1][0])
        datays.append(lane[-1][1])
        return np.array(dataxs, dtype=float), np.array(datays, dtype=float), False


if __name__ == '__main__':
    start = time.time()
    lane_math = LaneMath()
    # lane_math.show('/home/yuanning/DeepMotion/lane/data', True)  # 152 files
    sub = lane_math.show('/home/yuanning/DeepMotion/Hard-data/data', True)  # 410 files
    end = time.time()
    print("Time used for entire program: " + str(end - start))
    print("Subprogram used {}% of total time".format(100 * sub / (end - start)))
