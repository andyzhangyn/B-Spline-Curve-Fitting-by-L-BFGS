#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yuanning Zhang (yuanningzhang@deepmotion.ai)
# Copyright (c) 2018-present, DeepMotion


import numpy as np
import scipy as sp
import cubic_b_spline
import matplotlib.pyplot as plt
import scipy.optimize
import time


class BSplineModel(object):

    # class Point(object):
    #
    #     def __init__(self, x, y):
    #         self.x = x
    #         self.y = y
    #
    #     @staticmethod
    #     def dist_sqrt(p1, p2):
    #         dx = p1.x - p2.x
    #         dy = p1.y - p2.y
    #         return dx * dx + dy * dy

    def __init__(self, data, pts, img=None, id=None, eps=0.1):
        self.data = data
        self.pts = pts
        self.N = data.shape[0]  # fixed
        self.n = pts.shape[0]  # fixed
        self.cbs = cubic_b_spline.CubicBSpline(self.pts)
        self.T = np.zeros(self.N)
        k = (self.n - 3.0) / self.N
        for i in range(self.N):
            self.T[i] = 3 + i * k
        self.img = img
        self.id = id
        # self.eps = eps

    def l_bfgs_fitting(self):
        # last_min_val = 1000
        # min_val = 800
        # loss_history = {}
        A = np.transpose(self.pts)
        xs = A[0]
        ys = A[1]
        x = np.concatenate((xs, ys, self.T))
        # x = np.concatenate((xs, ys))
        x, min_val, info = sp.optimize.fmin_l_bfgs_b(self.loss, x, approx_grad=True, m=8, maxfun=1000)
        # for i in range(self.niter):
        #     # start = time.time()
        #     # if self.id is None:
        #     #     print("Iteration " + str(i))
        #     # else:
        #     #     print("Lane " + str(self.id) + ", Iteration " + str(i))
        #     if abs(last_min_val) < self.eps:
        #         print("Converge" + "\n")
        #         # end = time.time()
        #         # print("Time used: " + str(end - start) + "\n")
        #         self.plot_spline()
        #         return min_val, loss_history
        #     # start = time.time()
        #     x, min_val, info = sp.optimize.fmin_l_bfgs_b(self.loss, x, approx_grad=True, m=3, maxfun=2000)
        #     # end = time.time()
        #     # print("Time used: " + str(end - start) + "\n")
        #     if abs(last_min_val - min_val) < self.eps:
        #         print("Converge" + "\n")
        #         # end = time.time()
        #         # print("Time used: " + str(end - start) + "\n")
        #         self.plot_spline()
        #         return min_val, loss_history
        #     # print("Loss: " + str(min_val) + "\n")
        #     # end = time.time()
        #     # print("Time used: " + str(end - start) + "\n")
        #     last_min_val = min_val
        #     loss_history[i] = min_val
        #     for k in range(self.n):
        #         self.pts[k] = [x[k], x[k+self.n]]
        #     self.cbs = cubic_b_spline.CubicBSpline(self.pts)
        #     self.T = x[2*self.n:]
        #     # self.plot()
        # print("Max iterations reached\n")
        # end = time.time()
        # print("Time used: " + str(end - start) + "\n")
        # print("Lane finished")
        for k in range(self.n):
            self.pts[k] = [x[k], x[k+self.n]]
        self.cbs = cubic_b_spline.CubicBSpline(self.pts)
        # self.T = x[2*self.n:]
        self.plot_spline()
        # return min_val

    def loss(self, v):
        # start = time.time()
        newpts = np.zeros((self.n, 2))
        for k in range(self.n):
            newpts[k] = [v[k], v[k + self.n]]
        newcbs = cubic_b_spline.CubicBSpline(newpts)
        newT = v[2 * self.n:]
        # newT = self.T
        sum = 0
        for i in range(self.N):
            p1 = newcbs.eval(newT[i])
            p2 = self.data[i]
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            sum = sum + dx * dx + dy * dy
        # sum = sum / 2
        # sum = sum + self.fairing(newcbs, 3, self.n, 0.001, 2)
        # end = time.time()
        # print("Time used: " + str(end - start) + "\n")
        sum = sum * self.N
        return sum

    # @staticmethod
    # def fairing(cbs, a, b, alpha, precision):
    #     M = precision * (b - a)
    #     eps = 1.0 / precision
    #     sum = 0
    #     for i in range(M + 1):
    #         tplus = a + i * eps + eps / 2
    #         tminus = a + i * eps - eps / 2
    #         pts = cbs.generate(np.array([tminus, tplus]))
    #         p1 = pts[0]
    #         p2 = pts[1]
    #         xprime = (p1[0] - p2[0])
    #         yprime = (p1[1] - p2[1])
    #         sum = sum + xprime * xprime + yprime * yprime
    #     sum = sum / eps
    #     sum = sum * alpha / (b - a)
    #     return sum

    def plot(self):
        if self.img is not None:
            plt.imshow(self.img)
        xs = [point[0] for point in self.pts]
        ys = [point[1] for point in self.pts]
        plt.plot(xs, ys, "gs")
        newts = np.linspace(3, self.n, self.n * 10)
        newpts = self.cbs.generate(newts)
        newxs = [point[0] for point in newpts]
        newys = [point[1] for point in newpts]
        plt.plot(newxs, newys, "r")
        dataxs = [point[0] for point in self.data]
        datays = [point[1] for point in self.data]
        plt.plot(dataxs, datays, "b^")
        # plt.show()

    def plot_spline(self):
        # plt.imshow(self.img)
        newts = np.linspace(3, self.n, self.n * 10)
        newpts = self.cbs.generate(newts)
        newxs = [point[0] for point in newpts]
        newys = [point[1] for point in newpts]
        plt.plot(newxs, newys, "r")
        # plt.show()


if __name__ == '__main__':
    start = time.time()
    data_size = 30
    ctrlpts = np.array([[-1.5, 2.0], [-0.5, -1.5], [0.5, 2.0], [1.5, -1.0],
                        [2.5, 3.0], [3.5, -1.0], [4.5, 2.0]], dtype=np.float)
    dataxs = np.linspace(0, 3, data_size)
    datays = 2 * np.sin(dataxs*3) + 0.5
    np.random.seed(0)
    dataxs = dataxs + np.random.random(size=data_size) * 0.4 - 0.2
    datays = datays + np.random.random(size=data_size) * 0.4 - 0.2
    data = np.zeros((dataxs.shape[0], 2))
    for i in range(dataxs.shape[0]):
        data[i, 0] = dataxs[i]
        data[i, 1] = datays[i]
    model = BSplineModel(data, ctrlpts)
    model.plot()
    model.l_bfgs_fitting()
    plt.show()
    end = time.time()
    print("Time used: " + str(end - start))
