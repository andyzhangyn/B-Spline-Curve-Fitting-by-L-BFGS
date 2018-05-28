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


class LaneMath(object):

    def __init__(self):
        pass

    def show(self, data_path, axis=False):
        files = sorted(os.listdir(data_path))
        # k = 0
        # for file in files:
        #     lane_mask = np.load(os.path.join(data_path, file))
        #     self.show_graph(lane_mask, axis)
        #     print("File " + str(k) + " completed")
        #     k = k + 1
        lane_mask = np.load(os.path.join(data_path, files[4]))  # Error cases: 3,5,12,19,79
        self.show_graph(lane_mask, axis)

    def show_graph(self, lane_mask, axis=False):
        lane_mask = lane_mask[300:, 500:1400]
        img = self.colorise(lane_mask)
        plt.imshow(img)
        # plt.show()
        # plt.imshow(255 - 0 * img)
        if not axis:
            plt.axis('off')
        lane_ids = np.unique(lane_mask)
        lane_ids = lane_ids[1:]
        lanes = {}
        for id in lane_ids:
            lanes[id] = []
        for i in range(lane_mask.shape[0]):
            for j in range(lane_mask.shape[1]):
                if lane_mask[i, j] in lane_ids:
                    lanes[lane_mask[i, j]].append([j, i])
        for id in lane_ids:
            dataxs, datays = self.get_data_sample(lane_mask, id, lanes)
            # plt.plot(dataxs, datays, 'b^')
            n = dataxs.shape[0]
            data = np.zeros((n, 2))
            for i in range(n):
                data[i, 0] = dataxs[i]
                data[i, 1] = datays[i]
            numctrl = len(lanes[id]) / 800
            # arr = [[dataxs[0], datays[0]], [dataxs[0], datays[0]]]
            arr = [[dataxs[0], datays[0]], [dataxs[0], datays[0]], [dataxs[0], datays[0]]]
            # arr = [[dataxs[0], datays[0]], [dataxs[0], datays[0]]]
            for i in range(numctrl - 1):
                arr.append([dataxs[(i + 1) * n / numctrl], datays[(i + 1) * n / numctrl]])
                # plt.plot(dataxs[(i + 1) * n / numctrl], datays[(i + 1) * n / numctrl], 'ko')
                # arr.append([dataxs[(i + 1) * n / numctrl], datays[(i + 1) * n / numctrl]])
            arr.append([dataxs[n - 1], datays[n - 1]])
            arr.append([dataxs[n - 1], datays[n - 1]])
            arr.append([dataxs[n - 1], datays[n - 1]])
            ctrlpts = np.array(arr, dtype=np.float)
            # ctrlpts = np.array([[50, -50], [50, 50], [200, 100], [250, 300], [350, 500], [250, 600]], dtype=np.float)
            model = b_spline_model.BSplineModel(data, ctrlpts, img=img, id=id)
            # model.plot()
            model.l_bfgs_fitting()
        plt.show()

    @staticmethod
    def colorise(lane_mask, black_background=False):
        np.random.seed(0)
        palette = np.random.randint(70, 255, size=(100, 3), dtype=np.uint8)
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
        pca = PCA(n_components=1)
        data_new = pca.fit_transform(lanes[id])
        # print(pca.explained_variance_ratio_)
        extreme = pca.inverse_transform(min(data_new))
        # print(extreme)
        # print(pca.mean_)
        d1 = (extreme[0] - pca.mean_[0]) ** 2 + (extreme[1] - pca.mean_[1]) ** 2
        d2 = (lanes[id][0][0] - pca.mean_[0]) ** 2 + (lanes[id][0][1] - pca.mean_[1]) ** 2
        d3 = (lanes[id][-1][0] - pca.mean_[0]) ** 2 + (lanes[id][-1][1] - pca.mean_[1]) ** 2
        if d1 > d2 or d1 > d3 or pca.explained_variance_ratio_[0] > 0.998:
        # if d1 > d2 or d1 > d3:
        # if len(lanes[id]) < 1
            data_density = 200
            # pca = PCA(n_components=1)
            # pca.fit(lanes[id])
            lane = np.unique(data_new)
            lane = [[x] for x in lane]
            # print(lane)
            data = []
            # print(len(lane))
            for k in range(len(lane)):
                if k % data_density is 0:
                    # print("Hello" + str(lane[k]))
                    data.append(lane[k])
            data.append(lane[-1])
            data = np.array(data)
            newdata = pca.inverse_transform(data)
            # print(newdata)
            dataxs, datays = np.empty(newdata.shape[0]), np.empty(newdata.shape[0])
            for i in range(newdata.shape[0]):
                dataxs[i] = newdata[i, 0]
                datays[i] = newdata[i, 1]
            return np.array(dataxs), np.array(datays)

        dataxs, datays = [], []
        data_density = 200
        k = 0
        lane = lanes[id]
        if (abs(lane[0][1] - lane[-1][1]) + 1) / (abs(lane[0][0] - lane[-1][0]) + 1) < 1:
            for j in range(lane_mask.shape[1]):
                for i in range(lane_mask.shape[0]):
                    if lane_mask[i][j] == id:
                        if k is 0:
                            dataxs = [j]
                            datays = [i]
                        elif k % data_density is 0:
                            dataxs.append(j)
                            datays.append(i)
                        k = k + 1
        else:
            # for i in range(lane_mask.shape[0]):
            #     for j in range(lane_mask.shape[1]):
            #         if lane_mask[i][j] == id:
            #             if k is 0:
            #                 dataxs = [j]
            #                 datays = [i]
            #             elif k % data_density is 0:
            #                 dataxs.append(j)
            #                 datays.append(i)
            #             k = k + 1
            for pixel in lane:
                if k % data_density is 0:
                    dataxs.append(pixel[0])
                    datays.append(pixel[1])
                k = k + 1
        dataxs.append(lane[-1][0])
        datays.append(lane[-1][1])
        return np.array(dataxs), np.array(datays)


if __name__ == '__main__':
    start = time.time()
    lane_math = LaneMath()
    lane_math.show('/home/yuanning/DeepMotion/lane/data', True)
    end = time.time()
    print("Time used: " + str(end - start))
