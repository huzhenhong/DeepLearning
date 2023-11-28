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


def process(im):
    h, w, _ = im.shape
    roi = [0.642, 0.196, 0.658, 0.228]
    roi = [int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)]
    src = im[roi[1] : roi[3], roi[0] : roi[2]]

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU
    )

    # num_labels, labels = cv.connectedComponents(binary, connectivity=8, ltype=cv.CV_32S)
    # print(num_labels)  # output: 5

    # # 构造颜色
    # colors = []
    # for i in range(num_labels):
    #     b = np.random.randint(0, 256)
    #     g = np.random.randint(0, 256)
    #     r = np.random.randint(0, 256)
    #     colors.append((b, g, r))
    # colors[0] = (0, 0, 0)

    # # 画出连通图
    # h, w = gray.shape
    # image = np.zeros((h, w, 3), dtype=np.uint8)
    # for row in range(h):
    #     for col in range(w):
    #         image[row, col] = colors[labels[row, col]]
    # cv.imshow('image', image)


    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # morphology_open = cv.morphologyEx(
    #     binary, cv.MORPH_OPEN, kernel, iterations=1
    # )
    # morphology_close = cv.morphologyEx(
    #     morphology_open, cv.MORPH_CLOSE, kernel, iterations=1
    # )
    # cv.imshow('binary', np.hstack((binary, morphology_open, morphology_close)))
    # dilate = cv.dilate(binary, kernel, iterations=1)
    # erode = cv.erode(morphology_open, kernel, iterations=1)
    # cv.imshow('binary', np.hstack((binary, dilate, erode)))

    # hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    # # # cv.imshow('src', src)
    # # # cv.imshow('hsv', hsv)
    # # red_lower = np.array([0, 43, 46])
    # # red_upper = np.array([10, 255, 255])
    red_lower = np.array([156, 100, 46])
    red_upper = np.array([180, 255, 255])
    # # 156,100,46,180,255,255

    mask = cv.inRange(hsv, lowerb=red_lower, upperb=red_upper)
    cv.imshow('mask', mask)

    contours, _ = cv.findContours(
        binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    sorted_contours = sorted(
        contours, key=lambda c: cv.contourArea(c), reverse=True
    )
    # print('contours size: ', len(contours))

    # approx = cv.approxPolyDP(sorted_contours[0], 3.0, True)
    # draw = cv.drawContours(src.copy(), [approx], -1, (0, 255, 0), -1)

    draw = cv.drawContours(src.copy(), sorted_contours, 0, (0, 255, 0), -1)
    draw = cv.drawContours(draw, sorted_contours[1:], -1, (255, 255, 0), -1)

    # center, radius = cv.minEnclosingCircle(sorted_contours[0])
    # cv.circle(draw, (int(center[0]), int(center[0])), int(radius), (0, 255, 255), 1)

    cv.imshow('src_draw', np.hstack((src, draw)))

    new_img = np.zeros_like(binary)
    cv.drawContours(new_img, sorted_contours, 0, 255, -1)
    # cv.imshow('new_img', new_img)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # morphology_open = cv.morphologyEx(
    #     binary, cv.MORPH_OPEN, kernel, iterations=1
    # )
    # morphology_close = cv.morphologyEx(
    #     new_img, cv.MORPH_CLOSE, kernel, iterations=1
    # )
    # cv.imshow('binary', np.hstack((binary, morphology_open, morphology_close)))
    dilate = cv.dilate(new_img, kernel, iterations=2)
    erode = cv.erode(dilate, kernel, iterations=3)
    cv.imshow('new_img', np.hstack((new_img, dilate, erode)))

    light = cv.bitwise_and(src, src, mask=erode)
    cv.imshow('light', light)

    # cv.waitKey()
    # return np.hstack((new_img, dilate, erode))

    mean = cv.mean(src, mask=erode)
    print("mean: ", sum(mean))
    # minmax_b = cv.minMaxLoc(im, mask=mask)
    mv = cv.split(src)

    mean_b = int(mean[0])
    _, minmax_b, _, _ = cv.minMaxLoc(mv[0], mask=erode)
    minmax_b = int(minmax_b)

    mean_g = int(mean[1])
    _, minmax_g, _, _ = cv.minMaxLoc(mv[1], mask=erode)

    mean_r = int(mean[2])
    _, minmax_r, _, _ = cv.minMaxLoc(mv[2], mask=erode)
    minmax_r = int(minmax_r)

    # print(f'{mean_b} {mean_g} {mean_r} - {minmax_b} {minmax_g} {minmax_r}')

    if mean_r > 190 and minmax_r > 250:
        cv.putText(im, 'on', (roi[0], roi[1]), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        cv.putText(im, 'off', (roi[0], roi[1]), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    return im
    return None
