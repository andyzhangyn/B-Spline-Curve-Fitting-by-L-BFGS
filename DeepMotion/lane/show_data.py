#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yuanning Zhang (yuanningzhang@deepmotion.ai)
# Copyright (c) 2018-present, DeepMotion


import os
import cv2
import numpy as np
from lane_math import LaneMath


class ShowData(object):

    def __init__(self):
        pass

    def show(self, data_path):
        files = sorted(os.listdir(data_path))
        for fname in files:
            data = np.load(os.path.join(data_path, fname))
            cv2.imshow("Lane", LaneMath.colorise(data, True))
            cv2.waitKey(1)


if __name__ == '__main__':
    show_data = ShowData()
    show_data.show('/home/yuanning/DeepMotion/Hard-data/data')
