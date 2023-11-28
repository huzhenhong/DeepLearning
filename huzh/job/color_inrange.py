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


class ColorInRange:
    def __init__(self, im, color='orange') -> None:
        self.__color_table = {
            'black': {'min': [0, 0, 0], 'max': [180, 255, 46]},
            'gray': {'min': [0, 0, 46], 'max': [180, 43, 220]},
            'white': {'min': [0, 0, 221], 'max': [180, 30, 255]},
            'red': {'min': [0, 43, 46], 'max': [10, 255, 255]},
            'red1': {'min': [156, 43, 46], 'max': [180, 255, 255]},
            'orange': {'min': [11, 43, 46], 'max': [25, 255, 255]},
            'yellow': {'min': [26, 43, 46], 'max': [34, 255, 255]},
            'green': {'min': [35, 43, 46], 'max': [77, 255, 255]},
            'cyan': {'min': [78, 43, 46], 'max': [99, 255, 255]},
            'blue': {'min': [100, 43, 46], 'max': [124, 255, 255]},
            'purple': {'min': [125, 43, 46], 'max': [155, 255, 255]},
        }

        self.__im = im
        self.__color = color
        self.__hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)

        self.__h_min, self.__s_min, self.__v_min = np.array(
            self.__color_table[color]['min']
        )
        self.__h_max, self.__s_max, self.__v_max = np.array(
            self.__color_table[color]['max']
        )

        self.__h_low = self.__h_min
        self.__h_up = self.__h_max
        self.__s_low = self.__s_min
        self.__s_up = self.__s_max
        self.__v_low = self.__v_min
        self.__v_up = self.__v_max

        cv.namedWindow('in_range', cv.WINDOW_NORMAL)
        cv.resizeWindow('in_range', 600, 800)

        cv.createTrackbar(
            'h_min', 'in_range', self.__h_min, self.__h_max, self.h_min_callback
        )
        cv.createTrackbar(
            'h_max', 'in_range', self.__h_max, self.__h_max, self.h_max_callback
        )
        cv.createTrackbar(
            's_min', 'in_range', self.__s_min, self.__s_max, self.s_min_callback
        )
        cv.createTrackbar(
            's_max', 'in_range', self.__s_max, self.__s_max, self.s_max_callback
        )
        cv.createTrackbar(
            'v_min', 'in_range', self.__v_min, self.__v_max, self.v_min_callback
        )
        cv.createTrackbar(
            'v_max', 'in_range', self.__v_max, self.__v_max, self.v_max_callback
        )

        cv.setTrackbarMin('h_min', 'in_range', self.__h_min)
        cv.setTrackbarMin('h_max', 'in_range', self.__h_min)
        cv.setTrackbarMin('s_min', 'in_range', self.__s_min)
        cv.setTrackbarMin('s_max', 'in_range', self.__s_min)
        cv.setTrackbarMin('v_min', 'in_range', self.__v_min)
        cv.setTrackbarMin('v_max', 'in_range', self.__v_min)

    def h_min_callback(self, x):
        self.__h_low = x
        self.inrange()

    def h_max_callback(self, x):
        self.__h_up = x
        self.inrange()

    def s_min_callback(self, x):
        self.__s_low = x
        self.inrange()

    def s_max_callback(self, x):
        self.__s_up = x
        self.inrange()

    def v_min_callback(self, x):
        self.__v_low = x
        self.inrange()

    def v_max_callback(self, x):
        self.__v_up = x
        self.inrange()

    def inrange(self):
        print('min: ', [self.__h_low, self.__s_low, self.__v_low])
        print('max: ', [self.__h_up, self.__s_up, self.__v_up])

        lower = np.array([self.__h_low, self.__s_low, self.__v_low])
        upper = np.array([self.__h_up, self.__s_up, self.__v_up])

        mask = cv.inRange(self.__hsv, lowerb=lower, upperb=upper)
        cv.imshow('mask', mask)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL,  cv.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda c : cv.contourArea(c), reverse=True)
        draw_img = cv.drawContours(self.__im.copy(), sorted_contours, 0, (255, 0, 255), 1)

        if len(sorted_contours) > 1:
            draw_img = cv.drawContours(draw_img, sorted_contours[1:], -1, (255, 255, 0), 1)

        mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        cv.imshow('in_range', np.hstack((draw_img, mask)))


        # fill_img = np.zeros_like(self.__im)
        # fill_img += np.array((255, 0, 255), dtype=np.uint8)
        # fill_img = cv.bitwise_and(fill_img, fill_img, mask=mask)

        # draw_img = cv.addWeighted(self.__im, 0.5, fill_img, 0.5, 0)
        # cv.imshow('in_range', np.hstack((self.__im, draw_img)))


def process(im):
    ColorInRange(im, 'red')


if __name__ == '__main__':
    imgs_path = [
        str(path)
        for path in Path(
            '/Users/huzh/Documents/project/中车/指示灯开关检测/20231114/局放控制台灯20231109/灯亮'
            # '/Users/huzh/Documents/project/中车/指示灯开关检测/20231114/局放控制台灯20231109/灯灭'
        ).iterdir()
    ]

    for path in imgs_path:
        src = cv.imread(path)

        h, w, _ = src.shape
        roi = [0.642, 0.196, 0.658, 0.228]
        roi = [
            int(roi[0] * w),
            int(roi[1] * h),
            int(roi[2] * w),
            int(roi[3] * h),
        ]
        src = src[roi[1] : roi[3], roi[0] : roi[2]]

        # cv.imshow('src', src)
        process(src)
        cv.waitKey()
