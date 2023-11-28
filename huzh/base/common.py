# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2021-07-22 11:16:43
 LastEditors  : huzhenhong
 LastEditTime : 2021-07-31 15:35:10
 FilePath     : \\python\\base_library\\common.py
 Copyright (C) 2021 huzhenhong. All rights reserved.
'''

import os
import time
import numpy as np
from functools import wraps


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {} s'.format(func.__module__, func.__name__, end - start))
        return r
    return wrapper


def IOU(rect_1, rect_2):
    b1_x1, b1_y1, b1_x2, b1_y2 = rect_1[0], rect_1[1], rect_1[0] + rect_1[2], rect_1[1] + rect_1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = rect_2[0], rect_2[1], rect_2[0] + rect_2[2], rect_2[1] + rect_2[3]

    # Intersection area
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    if inter_y2 < inter_y1 or inter_x2 < inter_x1:
        return 0

    inter = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)

    # Union Area
    union_x1 = min(b1_x1, b2_x1)
    union_y1 = min(b1_y1, b2_y1)
    union_x2 = max(b1_x2, b2_x2)
    union_y2 = max(b1_y2, b2_y2)
    union = (union_y2 - union_y1) * (union_x2 - union_x1)

    return inter / union


def read_file(path):
    if not os.path.exists(path):
        return []

    f = open(path, 'r', encoding='utf-8')
    lines = f.read().splitlines()
    contents = [np.int0(line.strip().split(' ')) for line in lines]
    return contents
