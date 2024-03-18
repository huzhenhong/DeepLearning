# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 颜色提取
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2021-03-19 08:53:26
 LastEditors  : huzhenhong
 LastEditTime : 2021-03-25 17:34:25
 FilePath     : \\python\\003_light_status_detection\\src\\color_pick.py
 Copyright    : All rights reserved.
'''

import cv2 as cv
import numpy as np
from pathlib import Path


def process(im):
    hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)

    red_lower = np.array([0, 43, 46])
    red_upper = np.array([10, 255, 255])
    mask1 = cv.inRange(hsv, lowerb=red_lower, upperb=red_upper)

    red_lower = np.array([156, 100, 46])
    red_upper = np.array([180, 255, 255])
    mask2 = cv.inRange(hsv, lowerb=red_lower, upperb=red_upper)

    red_lower = np.array([11, 43, 46])
    red_upper = np.array([25, 255, 255])
    mask3 = cv.inRange(hsv, lowerb=red_lower, upperb=red_upper)

    red_lower = np.array([26, 43, 46])
    red_upper = np.array([34, 255, 255])
    mask4 = cv.inRange(hsv, lowerb=red_lower, upperb=red_upper)

    mask = cv.bitwise_or(mask1, mask2)
    mask = cv.bitwise_or(mask, mask3)
    mask = cv.bitwise_or(mask, mask4)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel=kernel)
    mask = cv.erode(mask, kernel=kernel)
    cv.imshow('inRange', mask)
    return mask
