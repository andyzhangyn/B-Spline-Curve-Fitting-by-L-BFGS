#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yuanning Zhang (yuanningzhang@deepmotion.ai)
# Copyright (c) 2018-present, DeepMotion


import os
import numpy as np


class ShowData(object):

    def __init__(self):
        pass

    def show(self, data_path):
        files = sorted(os.listdir(data_path))
        for fname in files:
            data = np.load(os.path.join(data_path, fname))
            print(data)

    def show_first(self, data_path):
        files = sorted(os.listdir(data_path))
        first_data = np.load(os.path.join(data_path, files[0]))
        # lane_ids = np.unique(first_data)
        # print(lane_ids)
        for item in first_data:
            for subitem in item:
                if subitem >= 0:
                    print(subitem)


if __name__ == '__main__':
    show_data = ShowData()
    show_data.show_first('/home/yuanning/DeepMotion/lane/data')
