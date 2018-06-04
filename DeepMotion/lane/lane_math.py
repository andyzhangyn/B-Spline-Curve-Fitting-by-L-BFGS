#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Kuiyuan Yang (kuiyuanyang@deepmotion.ai)
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
        # for k in range(0, 20):
        #     lane_mask = np.load(os.path.join(data_path, files[k]))
        #     self.show_graph(lane_mask, axis)
        # end = time.time()
        # print(end - start)
        lane_mask = np.load(os.path.join(data_path, files[17]))
        self.show_graph(lane_mask, axis)
        print(self.time)

    def show_graph(self, lane_mask, axis=False):
        total_time = 0.0
        # a = time.time()
        lane_mask = lane_mask[300:, 500:1400]
        # b = time.time()
        # plt.grid(True, linestyle="-.", color="r", linewidth="3")
        height = lane_mask.shape[0]
        width = lane_mask.shape[1]
        # a = time.time()
        img = self.colorise(lane_mask)
        # b = time.time()
        plt.imshow(img)
        # plt.show()
        # plt.imshow(255 - 0 * img)
        if not axis:
            plt.axis('off')
        lane_ids = np.unique(lane_mask)
        lane_ids = lane_ids[1:]
        lanes = {}
        lanes_trans = {}


        # A = np.transpose(lane_mask)
        # for id in lane_ids:
        #     lanes[id] = np.transpose(np.nonzero(A == id))
            # print(lanes[id])
            # xs, ys = [pixel[0] for pixel in lanes[id]], [pixel[1] for pixel in lanes[id]]
            # plt.plot(xs, ys, "r,")

        # for id in lane_ids:
        #     lanes[id] = []
        #     lanes_trans[id] = []
            # print(id)
        # a = time.time()
        # for i in range(lane_mask.shape[0]):
        #     for j in range(lane_mask.shape[1]):
        #         if lane_mask[i, j] in lane_ids:
        #             lanes[lane_mask[i, j]].append([j, i])
        A = np.transpose(lane_mask)
        B = lane_mask
        for id in lane_ids:
            lanes_trans[id] = np.transpose(np.nonzero(A == id))
            C = np.array(np.nonzero(B == id))
            C[[0, 1]] = C[[1, 0]]
            lanes[id] = np.transpose(C)
        #     print(lanes[id])
        #     xs, ys = [pixel[1] for pixel in lanes_trans[id]], [pixel[0] for pixel in lanes_trans[id]]
        #     plt.plot(xs, ys, "r,")
        # b = time.time()
        # print(b-a)
        # b = time.time()
        # a = time.time()
        # print(lane_ids)
        for id in lane_ids:
            a = time.time()
            # pca = PCA(n_components=1)
            # pca.fit(lanes[id])
            # if pca.explained_variance_ratio_[0] < 0.98 or True:
            #     lane = lanes[id]
            #     lanexs, laneys = [p[0] for p in lane], [p[1] for p in lane]
            #     lanexs, laneys = self.RANSAC_Quadratic_Interpolation(lanexs, laneys)
            #     plt.plot(lanexs, laneys, 'k,')
            #     lane = [[lanexs[k], laneys[k]] for k in range(len(lanexs))]
            #     lanes[id] = lane

            # lane = lanes[id]
            # lanexs, laneys = [p[0] for p in lane], [p[1] for p in lane]
            # lanexs, laneys = self.RANSAC_LeastSquare(lanexs, laneys)
            # # lanexs, laneys = self.RANSAC_LeastSquare(lanexs, laneys)
            # # plt.plot(lanexs, laneys, 'k,')
            # lane = [[lanexs[k], laneys[k]] for k in range(len(lanexs))]
            # lanes[id] = lane

            # lane = lanes[id]
            # sd = sort_data.SortData(lane, step_size=30)
            # lane = sd.getsorted()
            # lanexs, laneys = [p[0] for p in lane], [p[1] for p in lane]
            # lanexs, laneys = self.RANSAC_LeastSquare(lanexs, laneys)
            # lanexs, laneys = self.RANSAC_LeastSquare(lanexs, laneys)
            # plt.plot(lanexs, laneys, 'k,')
            # lane = [[lanexs[k], laneys[k]] for k in range(len(lanexs))]
            # lanes[id] = lane
            # lane = lanes[id]

            # a = time.time()
            if len(lanes[id]) < 500:
                continue

            # a = time.time()
            data = lanes[id]
            data_2 = lanes_trans[id]
            pca = PCA(n_components=1)
            newdata = pca.fit_transform(data)
            extreme = pca.inverse_transform(min(newdata))
            if abs(pca.components_[0][0]) > abs(pca.components_[0][1]):
                data = lanes_trans[id]
                data_2 = lanes[id]
            # print(extreme)
            # print(pca.mean_)
            d1 = (extreme[0] - pca.mean_[0]) ** 2 + (extreme[1] - pca.mean_[1]) ** 2
            d2 = (data[0][0] - pca.mean_[0]) ** 2 + (data[0][1] - pca.mean_[1]) ** 2
            d3 = (data[-1][0] - pca.mean_[0]) ** 2 + (data[-1][1] - pca.mean_[1]) ** 2
            d4 = (data_2[0][0] - pca.mean_[0]) ** 2 + (data_2[0][1] - pca.mean_[1]) ** 2
            d5 = (data_2[-1][0] - pca.mean_[0]) ** 2 + (data_2[-1][1] - pca.mean_[1]) ** 2
            # plt.plot([data[0][0]], [data[0][1]], "ro")
            # plt.plot([data[-1][0]], [data[-1][1]], "bo")
            # plt.plot([data_2[0][0]], [data_2[0][1]], "ro")
            # plt.plot([data_2[-1][0]], [data_2[-1][1]], "bo")
            # print(str(d1) + " " + str(d4) + " " + str(d5) + "\n")
            if pca.explained_variance_ratio_[0] > 0.998 or ((d1 > d4 or d1 > d5) and (d1 > d2 or d1 > d3)) or len(data) < 1000:
            # if pca.explained_variance_ratio_[0] > 0.99:
                mindata = min(newdata)
                maxdata = max(newdata)
                newdata = pca.inverse_transform([mindata, maxdata])

                newdataxs = np.array([newdata[0][0], newdata[-1][0]])
                newdatays = np.array([newdata[0][1], newdata[-1][1]])
                plt.plot(newdataxs, newdatays, "r")
                continue
            #
            # dataxs, datays, use_PCA = self.get_data_sample(lane_mask, id, lanes)
            # if use_PCA:
            #     dataxs = np.array([dataxs[0], dataxs[-1]])
            #     datays = np.array([datays[0], datays[-1]])
            #     plt.plot(dataxs, datays, "r")
            #     continue
            # plt.plot(dataxs, datays, 'b^')



            # data = [[dataxs[k], datays[k]] for k in range(len(dataxs))]
            # pca = PCA(n_components=1)
            # newdata = pca.fit_transform(data)
            # if pca.explained_variance_ratio_[0] > 0.99:
            #     newdata = pca.inverse_transform(newdata)
            #     newdataxs = np.array([newdata[0][0], newdata[-1][0]])
            #     newdatays = np.array([newdata[0][1], newdata[-1][1]])
            #     plt.plot(newdataxs, newdatays, "r")
            #     continue

            # dataxs, datays = [p[0] for p in data], [p[1] for p in data]
            A = np.transpose(data)
            dataxs, datays = A[0], A[1]
            # dataxs, datays = dataxs[::2], datays[::2]


            plt.plot(dataxs, datays, 'k,')
            # a = time.time()
            dataxs, datays = self.RANSAC_Quadratic_Interpolation(dataxs, datays, id=id)
            # b = time.time()
            # print(b-a)
            # plt.plot(dataxs, datays, 'k,')

            # print("Checkpoint1")

            # plt.plot(dataxs, datays, 'b^')

            # a = time.time()
            sd = sort_data.SortData([[dataxs[k], datays[k]] for k in range(len(dataxs))], step_size=30, tolerance=50)
            data = sd.getsorted()
            # b = time.time()
            plt.plot([p[0] for p in data], [p[1] for p in data], 'rs')
            A = np.transpose(data)
            dataxs, datays = A[0], A[1]
            # plt.imshow(255 - np.zeros((lane_mask.shape[0], lane_mask.shape[1], 3)))
            # plt.show()

            # print("Checkpoint2")

            pca = PCA(n_components=1)
            newdata = pca.fit_transform(data)
            if pca.explained_variance_ratio_[0] > 0.997:
                newdata = pca.inverse_transform(newdata)
                newdataxs = np.array([newdata[0][0], newdata[-1][0]])
                newdatays = np.array([newdata[0][1], newdata[-1][1]])
                plt.plot(newdataxs, newdatays, "r")
                continue

            # lane = lanes[id]
            # lanexs, laneys = [p[0] for p in lane], [p[1] for p in lane]
            # lanexs, laneys = self.RANSAC_Quadratic_Interpolation(lanexs, laneys, id=id)
            # # lanexs, laneys = self.RANSAC_LeastSquare(lanexs, laneys)
            # plt.plot(lanexs, laneys, 'k,')
            # lane = [[lanexs[k], laneys[k]] for k in range(len(lanexs))]
            # lanes[id] = lane

            # dataxs, datays, use_PCA = self.get_data_sample(lane_mask, id, lanes)
            # if use_PCA:
            #     plt.plot(dataxs, datays, "r")
            #     continue

            # dataxs, datays = self.RANSAC_LeastSquare(dataxs, datays)
            # dataxs, datays = self.RANSAC_Quadratic_Interpolation(dataxs, datays)
            # plt.plot(dataxs, datays, 'b^')
            # a = time.time()
            n = len(dataxs)
            # data = np.array(zip(dataxs, datays))
            data = np.zeros((n, 2))
            for i in range(n):
                data[i, 0] = dataxs[i]
                data[i, 1] = datays[i]

            # b = time.time()
            numctrl = len(lanes[id]) / 800
            # print(dataxs)
            # print(datays)
            arr = [[dataxs[0], datays[0]], [dataxs[0], datays[0]], [dataxs[0], datays[0]]]
            # arr = [[data[0, 0], data[0, 1]], [data[0, 0], data[0, 1]], [data[0, 0], data[0, 1]]]
            # arr = [[dataxs[0], datays[0]], [dataxs[0], datays[0]]]
            # a = time.time()
            for i in range(numctrl - 1):
                arr.append([dataxs[(i + 1) * n / numctrl], datays[(i + 1) * n / numctrl]])
                # plt.plot(dataxs[(i + 1) * n / numctrl], datays[(i + 1) * n / numctrl], 'ko')
                # arr.append([dataxs[(i + 1) * n / numctrl], datays[(i + 1) * n / numctrl]])
            # b = time.time()
            arr.append([dataxs[n - 1], datays[n - 1]])
            arr.append([dataxs[n - 1], datays[n - 1]])
            arr.append([dataxs[n - 1], datays[n - 1]])
            # arr.append([data[n - 1][0], data[n - 1][1]])
            # arr.append([data[n - 1][0], data[n - 1][1]])
            ctrlpts = np.array(arr, dtype=np.float)
            # ctrlpts = np.array([[50, -50], [50, 50], [200, 100], [250, 300], [350, 500], [250, 600]], dtype=np.float)

            model = b_spline_model.BSplineModel(data, ctrlpts, img=img, id=id)

            # model.plot()

            # a = time.time()
            model.l_bfgs_fitting()
            b = time.time()
            self.time = self.time + b - a
            # print(b - a)
            # total_time = total_time + b - a
        plt.show()
        # self.time = self.time + total_time
        # b = time.time()
        # self.time = self.time + b - a
        # print("Total time used: " + str(total_time))

    def RANSAC_Quadratic_Interpolation(self, dataxs, datays, niter=50, threshold=12, d=1000, id=None):
        if len(dataxs) < 4:
            return dataxs, datays
        np.random.seed(0)
        # best_dataxs = dataxs
        # best_datays = datays
        # min_loss = 100000000
        max_gain = 0
        n = len(dataxs)
        best_polyx = None
        best_polyy = None
        # if id == 1:
        #     niter = 30
        for i in range(niter):
            # if id is not None:
            #     print("Lane " + str(id) + ", Iteration " + str(i))
            # else:
            #     print("Iteration " + str(i))
            newdataxs = []
            newdatays = []
            # print(n)
            rand1 = 0
            rand2 = 0
            rand3 = 0
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
            #
            # rand1 = np.random.randint(0, n / 4)
            # rand2 = np.random.randint(n / 4, 3 * n / 4)
            # rand3 = np.random.randint(3 * n / 4, n)
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
            # ts = np.linspace(-1, 3, 40)
            # plt.plot([dataxs[rand1], dataxs[rand2], dataxs[rand3]], [datays[rand1], datays[rand2], datays[rand3]], "bs")
            # plt.plot(polyx(ts), polyy(ts), "k")
            # loss = 0
            gain = 0
            M = 20
            for j in range(len(dataxs)):
                p3 = np.array([dataxs[j], datays[j]])

                A = np.linspace(-1, 3, M * 4)
                B = np.transpose(np.array([polyx(A), polyy(A)]))
                min_point2curve = min(min(distance.cdist([p3], B)))
                # print(min_point2curve)

                # min_point2curve = 100000000
                # for k in range(0, 4 * M + 1):
                #     K = k / float(M) - 1
                #     point2curve = (polyx(K) - p3[0]) ** 2 + (polyy(K) - p3[1]) ** 2
                #     if point2curve < min_point2curve:
                #         min_point2curve = point2curve
                # min_point2curve = math.sqrt(min_point2curve)


                # if min_point2curve > threshold:
                #     min_point2curve = 2 * threshold
                if min_point2curve <= threshold:
                    newdataxs.append(p3[0])
                    newdatays.append(p3[1])
                    gain = gain + 1
                # else:
                #     # loss = loss + min_point2curve ** 2
                #     loss = loss + 1
            newdataxs = np.array(newdataxs)
            newdatays = np.array(newdatays)
            if len(newdataxs) >= 5 * n / 6:
                return newdataxs, newdatays
            # if loss < min_loss:
            #     min_loss = loss
            #     best_dataxs = newdataxs
            #     best_datays = newdatays
                best_polyx = polyx
                best_polyy = polyy
            if gain > max_gain:
                max_gain = gain
                best_dataxs = newdataxs
                best_datays = newdatays
                best_polyx = polyx
                best_polyy = polyy
            if gain == n:
                break
        print("Hello")
        ts = np.linspace(-1, 3, 40)
        if best_polyx is None or best_polyy is None:
            print("Lane " + str(id) + " RANSAC failed")
        else:
            plt.plot(best_polyx(ts), best_polyy(ts), "k")
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
            # p2 = data(np.random.randint(0, len(data)))
            v = p1 - p2
            loss = 0
            for j in range(len(dataxs)):
                p3 = np.array([dataxs[j], datays[j]])
                w = p3 - p2
                sq_dist = w[0] ** 2 + w[1] ** 2 - (v[0] * w[0] + v[1] * w[1]) ** 2 / (v[0] ** 2 + v[1] ** 2)
                if sq_dist < 0:
                    sq_dist = 0
                loss = loss + sq_dist
                # print(v)
                # print(w)
                # print(sq_dist)
                dist = math.sqrt(sq_dist)
                if dist <= threshold:
                    # print("Hey")
                    # print(p3)
                    # np.append(newdataxs, p3[0])
                    # np.append(newdatays, p3[1])
                    newdataxs.append(p3[0])
                    newdatays.append(p3[1])
                    # print(newdatays)
            # print(newdataxs)
            # print(newdatays)
            newdataxs = np.array(newdataxs)
            newdatays = np.array(newdatays)
            if len(newdataxs) >= d:
                # print("Hello")
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
        # n = len(lane)
        # for i in range(n / data_density):
        #     k = np.random.randint(n)
        #     dataxs.append(lane[k][0])
        #     datays.append(lane[k][1])
        while k < len(lane):
            dataxs.append(lane[k][0])
            datays.append(lane[k][1])
            k = k + data_density
        dataxs.append(lane[-1][0])
        datays.append(lane[-1][1])


        # data = [[dataxs[k], datays[k]] for k in range(len(dataxs))]
        #
        # pca = PCA(n_components=1)
        # # data_new = pca.fit_transform(lanes[id])
        # data_new = pca.fit_transform(data)
        # # print(pca.explained_variance_ratio_)
        # extreme = pca.inverse_transform(min(data_new))
        # # print(extreme)
        # # print(pca.mean_)
        # d1 = (extreme[0] - pca.mean_[0]) ** 2 + (extreme[1] - pca.mean_[1]) ** 2
        # d2 = (data[0][0] - pca.mean_[0]) ** 2 + (data[0][1] - pca.mean_[1]) ** 2
        # d3 = (data[-1][0] - pca.mean_[0]) ** 2 + (data[-1][1] - pca.mean_[1]) ** 2
        # if d1 > d2 or d1 > d3 or pca.explained_variance_ratio_[0] > 0.9992 or len(lanes[id]) < 1000:
        # # if pca.explained_variance_ratio_[0] > 0.998 or len(lanes[id]) < 1000:
        # # if d1 > d2 or d1 > d3:
        # # if len(lanes[id]) < 1
        #     data_density = 30
        #     # pca = PCA(n_components=1)
        #     # pca.fit(lanes[id])
        #     lane = np.unique(data_new)
        #     lane = [[x] for x in lane]
        #     # print(lane)
        #     data = []
        #     # print(len(lane))
        #     for k in range(len(lane)):
        #         if k % data_density is 0:
        #             # print("Hello" + str(lane[k]))
        #             data.append(lane[k])
        #     data.append(lane[-1])
        #     data = np.array(data)
        #     newdata = pca.inverse_transform(data)
        #     # print(newdata)
        #     dataxs, datays = np.empty(newdata.shape[0]), np.empty(newdata.shape[0])
        #     for i in range(newdata.shape[0]):
        #         dataxs[i] = newdata[i, 0]
        #         datays[i] = newdata[i, 1]
        #     return np.array(dataxs), np.array(datays), True


        # dataxs, datays = [], []
        # data_density = 100
        # k = 0
        # lane = lanes[id]
        # while k < len(lane):
        #     dataxs.append(lane[k][0])
        #     datays.append(lane[k][1])
        #     k = k + data_density

        # if abs(pca.components_[0][1]) / (abs(pca.components_[0][0]) + 0.1) < 1:
        # # if (abs(lane[0][1] - lane[-1][1]) + 1) / (abs(lane[0][0] - lane[-1][0]) + 1) < 1:
        #     for j in range(lane_mask.shape[1]):
        #         for i in range(lane_mask.shape[0]):
        #             if lane_mask[i][j] == id:
        #                 if k is 0:
        #                     dataxs = [j]
        #                     datays = [i]
        #                 elif k % data_density is 0:
        #                     dataxs.append(j)
        #                     datays.append(i)
        #                 k = k + 1
        # else:
        #     # for i in range(lane_mask.shape[0]):
        #     #     for j in range(lane_mask.shape[1]):
        #     #         if lane_mask[i][j] == id:
        #     #             if k is 0:
        #     #                 dataxs = [j]
        #     #                 datays = [i]
        #     #             elif k % data_density is 0:
        #     #                 dataxs.append(j)
        #     #                 datays.append(i)
        #     #             k = k + 1
        #     for pixel in lane:
        #         if k % data_density is 0:
        #             dataxs.append(pixel[0])
        #             datays.append(pixel[1])
        #         k = k + 1
        # dataxs.append(lane[-1][0])
        # datays.append(lane[-1][1])

        # n = len(dataxs)
        # data = np.zeros((n, 2))
        # for i in range(n):
        #     data[i, 0] = dataxs[i]
        #     data[i, 1] = datays[i]
        return np.array(dataxs, dtype=float), np.array(datays, dtype=float), False


if __name__ == '__main__':
    start = time.time()
    lane_math = LaneMath()
    # start = time.time()
    # lane_math.show('/home/yuanning/DeepMotion/lane/data', True)  # 152 files
    lane_math.show('/home/yuanning/DeepMotion/Hard-data/data', True)  # 410 files
    end = time.time()
    print("Time used: " + str(end - start))
