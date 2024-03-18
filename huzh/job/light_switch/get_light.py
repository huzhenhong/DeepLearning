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


pre_light_mask = []
pre_radius = []
invalid_cnt = 0

# def preprocess(binary):
#     gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
#     _, binary = cv.threshold(
#         gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU
#     )

#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
#     # morph_open = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
#     morph_close = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
#     # # dilate = cv.dilate(light_mask, kernel, iterations=2)
#     # # erode = cv.erode(dilate, kernel, iterations=3)
#     # cv.imshow('binary', np.hstack((binary, morph_open, morph_close)))

#     return morph_close


def find_light_mask(binary, src):
    contours, hieracy = cv.findContours(
        # binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        binary,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
    )

    im_h, im_w = binary.shape[:2]
    valid_contour = []

    for c in contours:
        area = cv.contourArea(c)
        if area < 20:
            continue

        center, radius = cv.minEnclosingCircle(c)

        if (
            center[0] + radius > im_w
            or center[0] - radius < 0
            or center[1] + radius > im_h
            or center[1] - radius < 0
        ):
            # print('invalid contour')
            continue

        valid_contour.append(c)

    if len(valid_contour) == 0:
        print('no contour')
        return None, 0

    sorted_contours = sorted(
        valid_contour, key=lambda c: cv.contourArea(c), reverse=True
    )
    light_contour = sorted_contours[0]

    center, radius = cv.minEnclosingCircle(light_contour)
    radius = int(radius * 0.8)
    cv.circle(
        src, (int(center[0]), int(center[1])), int(radius), (255, 0, 255), 1
    )
    # cv.imshow('minEnclosingCircle', src)

    # bbox = cv.boundingRect(c)
    # x1, y1, w, h = bbox
    # x2 = x1 + w
    # y2 = y1 + h
    # if x1 < 1 or y1 < 1 or src.shape[1] - x2 < 1 or src.shape[0] - y2 < 1:
    #     # print('invalid contour')
    #     # return None
    #     continue
    light_mask = np.zeros_like(binary)
    cv.circle(
        light_mask,
        (int(center[0] + 0.5), int(center[1] + 0.5)),
        int(radius + 0.5),
        255,
        -1,
    )
    # cv.drawContours(light_mask, [light_contour], 0, 255, -1)
    return light_mask, radius


def process(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU
    )
    cv.imshow('binary', binary)

    # 先正常查找一次
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    morph_close = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    cv.imshow('morph_close', morph_close)
    light_mask, radius = find_light_mask(morph_close, src)

    # 二值取反再尝试查找
    if light_mask is None:
        print('morph_close can not find light mask')
        binary_inv = cv.bitwise_not(binary)
        morph_open = cv.morphologyEx(
            binary_inv, cv.MORPH_OPEN, kernel, iterations=2
        )
        cv.imshow('morph_open', morph_open)
        light_mask, radius = find_light_mask(morph_open, src)
    else:
        cv.destroyWindow('morph_open')

    # 位置、大小是否突变
    is_valid = True

    if light_mask is not None:
        if len(pre_light_mask) > 0:
            final_pre_mask = None
            for pre_mask in pre_light_mask:
                if final_pre_mask is None:
                    final_pre_mask = pre_mask
                else:
                    final_pre_mask = cv.bitwise_or(final_pre_mask, pre_mask)

            and_mask = cv.bitwise_and(final_pre_mask, light_mask)
            iou = cv.countNonZero(and_mask) / (cv.countNonZero(light_mask) + 1)
            if iou < 0.5:
                print('wrong position or size')
                is_valid = False
        # if len(pre_radius) > 0:
        #     mean_radius = 1.0 * sum(pre_radius) / len(pre_radius)
        #     if abs(mean_radius - radius) / mean_radius > 0.3:
        #         print('mean_radius: ', mean_radius)
        #         print('radius: ', radius)
        #         print('wrong size')
        #         is_valid = False
        else:
            print('firt frame')
    else:
        print('morph_open can not find light mask')
        is_valid = False

    if len(pre_light_mask) > 5:
        pre_light_mask.pop()
    # if len(pre_radius) > 5:
    #     pre_radius.pop()

    if not is_valid:
        global invalid_cnt
        invalid_cnt += 1
        if invalid_cnt > 3:
            invalid_cnt = 0
            pre_light_mask.clear()
            # pre_radius.clear()

        # pre_light_mask.append(pre_light_mask[-1])
        # pre_radius.append(pre_radius[-1])
        return None, None

    pre_light_mask.append(light_mask)
    # pre_radius.append(radius)

    # cv.imshow('light_mask', light_mask)
    light = cv.bitwise_and(src, src, mask=light_mask)
    # cv.imshow('light', light)
    return light, light_mask

    cv.imshow('binary', binary)

    light_mask, binary = find_light_mask(
        src, cv.THRESH_BINARY_INV | cv.THRESH_OTSU
    )
    if light_mask is None:
        cv.imshow('binary_1', binary)
        light_mask, binary = find_light_mask(
            src, cv.THRESH_BINARY | cv.THRESH_OTSU
        )
        cv.imshow('binary_2', binary)

    # 没找到灯区域
    if light_mask is None:
        # print('no light found')

        zero_mask = np.zeros_like(src, np.uint8)
        cv.imshow('light_mask', np.hstack((zero_mask, zero_mask, zero_mask)))
        # cv.imshow('light_circle_bin', zero_mask)

        # zero_light = np.zeros_like(src)
        # cv.imshow('minEnclosingCircle', zero_light)
        # cv.imshow('light', zero_light)

        return None, None

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dilate = cv.morphologyEx(light_mask, cv.MORPH_CLOSE, kernel)
    erode = cv.morphologyEx(light_mask, cv.MORPH_OPEN, kernel)
    # dilate = cv.dilate(light_mask, kernel, iterations=2)
    # erode = cv.erode(dilate, kernel, iterations=3)
    cv.imshow('light_mask', np.hstack((light_mask, dilate, erode)))

    # # close_bin = cv.morphologyEx(light_mask, cv.MORPH_CLOSE, kernel)
    # close_bin = dilate
    # contours, hieracy = cv.findContours(
    #     close_bin,
    #     cv.RETR_EXTERNAL,
    #     cv.CHAIN_APPROX_SIMPLE,
    # )

    # sorted_contours = sorted(
    #     contours, key=lambda c: cv.contourArea(c), reverse=True
    # )

    # center, radius = cv.minEnclosingCircle(sorted_contours[0])
    # radius = int(radius * 0.8)
    # cv.circle(
    #     src, (int(center[0]), int(center[1])), int(radius), (0, 255, 255), 1
    # )
    # cv.imshow('minEnclosingCircle', src)

    # light_circle_mask = np.zeros_like(light_mask)
    # light_circle_mask = cv.circle(
    #     light_circle_mask,
    #     (int(center[0]), int(center[1])),
    #     int(radius),
    #     255,
    #     -1,
    # )
    # light_circle_color = cv.bitwise_and(src, src, mask=light_circle_mask)
    # light_circle_bin = cv.cvtColor(light_circle_color, cv.COLOR_BGR2GRAY)
    # _, light_circle_bin = cv.threshold(
    #     light_circle_bin, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU
    # )
    # cv.imshow('light_circle_bin', light_circle_bin)

    # light = cv.bitwise_and(src, src, mask=light_circle_mask)
    # cv.imshow('light_circle_mask', light_circle_mask)

    # cv.imshow('light', light)

    # print("radius: ", radius)
    # if int(radius) < 5:
    #     print('radius to small')
    #     return None, None

    # return light, light_circle_mask
    return None, None

    # h, w, _ = im.shape
    # roi = [0.642, 0.196, 0.658, 0.228]
    # roi = [int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)]
    # src = im[roi[1] : roi[3], roi[0] : roi[2]]

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(
        gray,
        0,
        255,
        cv.THRESH_BINARY | cv.THRESH_OTSU
        # gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU
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
    # red_lower = np.array([156, 100, 46])
    # red_upper = np.array([180, 255, 255])
    # # 156,100,46,180,255,255

    # mask = cv.inRange(hsv, lowerb=red_lower, upperb=red_upper)
    # cv.imshow('mask', mask)

    contours, hieracy = cv.findContours(
        # binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        binary,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
    )
    print('hieracy: ', hieracy)

    for c, h in zip(contours, hieracy[0]):
        if h[-1] != -1:
            r = np.random.randint(0, 255)
            g = np.random.randint(0, 255)
            b = np.random.randint(0, 255)
            draw = cv.drawContours(src, [c], 0, (b, g, r), 1)

    sorted_contours = sorted(
        contours, key=lambda c: cv.contourArea(c), reverse=True
    )
    # print('contours size: ', len(contours))

    # approx = cv.approxPolyDP(sorted_contours[0], 3.0, True)
    # draw = cv.drawContours(src.copy(), [approx], -1, (0, 255, 0), -1)

    draw = cv.drawContours(src.copy(), sorted_contours, 0, (0, 255, 0), 1)
    draw = cv.drawContours(draw, sorted_contours[1:], -1, (255, 255, 0), 1)

    # center, radius = cv.minEnclosingCircle(sorted_contours[0])
    # cv.circle(draw, (int(center[0]), int(center[0])), int(radius), (0, 255, 255), 1)

    # cv.imshow('src_draw', np.hstack((src, draw)))

    new_img = np.zeros_like(binary)

    max_contour = sorted_contours[0]
    x1, y1, w, h = cv.boundingRect(max_contour)
    x2 = x1 + w
    y2 = y1 + h
    if (
        x1 > 10
        and y1 > 10
        and src.shape[1] - x2 > 10
        and src.shape[0] - y2 > 10
    ):
        cv.drawContours(new_img, [c], 0, 255, -1)
    else:
        print('error binary: ', x1, y1, x2, y2)
    # for c in sorted_contours:
    #     x1, y1, w, h = cv.boundingRect(c)
    #     x2 = x1 + w
    #     y2 = y1 + h
    #     if x1 < 1 or y1 < 1 or src.shape[1] - x2 < 1 or src.shape[0] - y2 < 1:
    #         print('error binary')
    #         continue
    #     cv.drawContours(new_img, [c], 0, 255, -1)
    #     break

    # cv.imshow('new_img', new_img)

    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # # morphology_open = cv.morphologyEx(
    # #     binary, cv.MORPH_OPEN, kernel, iterations=1
    # # )
    # # morphology_close = cv.morphologyEx(
    # #     new_img, cv.MORPH_CLOSE, kernel, iterations=1
    # # )
    # # cv.imshow('binary', np.hstack((binary, morphology_open, morphology_close)))
    # dilate = cv.dilate(new_img, kernel, iterations=2)
    # erode = cv.erode(dilate, kernel, iterations=3)
    # cv.imshow('new_img', np.hstack((binary, new_img, dilate, erode)))

    # light = cv.bitwise_and(src, src, mask=erode)
    # cv.imshow('light', light)

    # center, radius = cv.minEnclosingCircle(sorted_contours[0])
    # cv.circle(draw, (int(center[0]), int(center[0])), int(radius), (0, 255, 255), 1)
    # cv.imshow('minEnclosingCircle', draw)

    # return light, erode
    # # return np.hstack((new_img, dilate, erode))

    # mean = cv.mean(src, mask=erode)
    # print("mean: ", sum(mean))
    # # minmax_b = cv.minMaxLoc(im, mask=mask)
    # mv = cv.split(src)

    # mean_b = int(mean[0])
    # _, minmax_b, _, _ = cv.minMaxLoc(mv[0], mask=erode)
    # minmax_b = int(minmax_b)

    # mean_g = int(mean[1])
    # _, minmax_g, _, _ = cv.minMaxLoc(mv[1], mask=erode)

    # mean_r = int(mean[2])
    # _, minmax_r, _, _ = cv.minMaxLoc(mv[2], mask=erode)
    # minmax_r = int(minmax_r)

    # # print(f'{mean_b} {mean_g} {mean_r} - {minmax_b} {minmax_g} {minmax_r}')

    # if mean_r > 190 and minmax_r > 250:
    #     cv.putText(im, 'on', (roi[0], roi[1]), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    # else:
    #     cv.putText(im, 'off', (roi[0], roi[1]), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # return im
    # return None
