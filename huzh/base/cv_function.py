# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : opencv基础函数
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2021-03-16 11:38:32
 LastEditors  : huzhenhong
 LastEditTime : 2021-03-30 10:10:37
 FilePath     : \\python\\base_library\\cv_function.py
 Copyright    : All rights reserved.
'''

import os
import cv2 as cv
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt


def waitKey(key=27):
    while True:
        if key == cv.waitKeyEx():
            break
        time.sleep(0.1)


def show(win_name, show_img, is_wait=False):
    """
    opencv 显示图片
    :param win_name: 窗口名称
    :param show_img: 待显示图片
    :return: 无
    """
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, show_img)

    if is_wait is True:
        cv.waitKey()


def resize(src_img, width=None, height=None, inter=cv.INTER_AREA):
    """
    调整图像尺寸
    :param src_img: 原图
    :param width: 调整后的宽
    :param height: 调整后的高
    :param inter: 插值方法
    :return: 调整后的图像
    """
    if width is None and height is None:
        return src_img  # 不予变换

    h, w = src_img.shape[:2]
    if h <= 0 or w <= 0:
        return src_img

    if width is None and height > 0:
        rate = float(h) / height
        return cv.resize(src_img, (int(w / rate + 0.5), height), interpolation=inter)

    elif height is None and width > 0:
        rate = float(w) / width
        return cv.resize(src_img, (width, int(h / rate + 0.5)), interpolation=inter)

    else:
        return cv.resize(src_img, (width, height), interpolation=inter)


def rotate(src_img, angle, scale=1):
    """
    :param src_img:
    :param angle:
    :return:
    """
    rows, cols = src_img.shape[:2]
    M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, scale)
    rotated_img = cv.warpAffine(src_img, M, (cols, rows))
    return rotated_img


def calc_x_angle(start_point, end_point):
    '''
    0------------------> x
    |           C
    |
    |       A       B
    v
    y

    AB 即为 x 轴正方向
    cos(A) = (AC * AB) / (|AC| * |AB|)
    A = arcose(cos(A)) * 180 / PI
    '''
    # A : start_point
    # C : end_point
    if start_point[1] < end_point[1]:
        start_point, end_point = end_point, start_point

    AB = np.array([1, 0])
    AC = np.array([end_point[0] - start_point[0], end_point[1] - start_point[1]])

    cos_angle = AB.dot(AC) / (np.sqrt(AB.dot(AB)) * np.sqrt(AC.dot(AC)))
    return np.arccos(cos_angle) * 180 / np.pi
    # return np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0]) * (180 / np.pi)


def get_roi_points_by_contour(contour, border=10, is_minRect=True):
    """
    :param contour:
    :param border:
    :is_minRect:
    :return: 四个拐点坐标，整型
    """

    src_points = np.zeros((4, 2), dtype=np.float32)       # 矩形四个顶点的坐标

    if is_minRect is True:
        # 最小外接斜矩形
        min_bounding_rect = cv.minAreaRect(contour)     # (圆心，长宽，旋转角度)

        # 四个顶点逆时针排序，第一个为与x轴交点
        points = cv.boxPoints(min_bounding_rect)

        # 对这四个点按x坐标排序，分为左边点和右边点
        points_sorted_by_x = sorted(points, key=lambda p: p[0])   # 从小到大
        left_points = points_sorted_by_x[:2]
        right_points = points_sorted_by_x[2:]

        # 对左边两个点按y坐标排序
        left_points_sorted_by_y = sorted(left_points, key=lambda p: p[1])   # 从小到大
        src_points[0] = left_points_sorted_by_y[0]
        src_points[3] = left_points_sorted_by_y[1]

        # 对右边两个点按y坐标排序
        right_points_sorted_by_y = sorted(right_points, key=lambda p: p[1])   # 从小到大
        src_points[1] = right_points_sorted_by_y[0]
        src_points[2] = right_points_sorted_by_y[1]
    else:
        # 外接矩形，四个点顺时针排列，起始点为左上角，数值为浮点数，但是是整数
        x, y, w, h = cv.boundingRect(contour)
        src_points[0] = [x, y]
        src_points[1] = [x + w, y]
        src_points[2] = [x + w, y + h]
        src_points[3] = [x, y + h]

    # 抠图加上border
    src_points[0] = src_points[0] + [-border, -border]
    src_points[1] = src_points[1] + [border, -border]
    src_points[2] = src_points[2] + [border, border]
    src_points[3] = src_points[3] + [-border, border]

    return src_points.astype(np.int32)


def perspective_transform(src, src_points):
    if src is None or len(src_points) != 4:
        return

    # A----------B
    # |          |
    # C----------D
    # 抠图宽高，因为透视变换要求整数，float强转int32
    # 向量 AB
    AB = src_points[1] - src_points[0]
    CD = src_points[2] - src_points[3]
    AC = src_points[3] - src_points[0]
    BD = src_points[2] - src_points[1]

    dst_width = int(max(np.sqrt(AB.dot(AB)), np.sqrt(CD.dot(CD))))
    dst_height = int(max(np.sqrt(AC.dot(AC)), np.sqrt(BD.dot(BD))))

    dst_points = np.array([[0, 0], [dst_width - 1, 0], [dst_width - 1, dst_height - 1], [0, dst_height - 1]], dtype=np.float32)
    M = cv.getPerspectiveTransform(src_points, dst_points)

    return cv.warpPerspective(src, M, (dst_width, dst_height))


def get_inpaint_mask(size, compose_points, border=1):
    """
    :param size: 掩膜大小
    :param compose_points: 轮廓点
    :border:轮廓边框，必须为非负
    :return: 单通道掩膜
    """
    # 确定掩膜
    out_frame_points = []
    out_frame_points.append((compose_points[0][0] - border, compose_points[0][1] - border))
    out_frame_points.append((compose_points[1][0] + border, compose_points[1][1] - border))
    out_frame_points.append((compose_points[2][0] + border, compose_points[2][1] + border))
    out_frame_points.append((compose_points[3][0] - border, compose_points[3][1] + border))
    out_frame_points = np.array(out_frame_points, np.int32)

    inner_frame_points = []
    border = -border
    inner_frame_points.append((compose_points[0][0] - border, compose_points[0][1] - border))
    inner_frame_points.append((compose_points[1][0] + border, compose_points[1][1] - border))
    inner_frame_points.append((compose_points[2][0] + border, compose_points[2][1] + border))
    inner_frame_points.append((compose_points[3][0] - border, compose_points[3][1] + border))
    inner_frame_points = np.array(inner_frame_points, np.int32)

    inpaint_mask = np.zeros(size, dtype=np.uint8)
    cv.fillConvexPoly(inpaint_mask, out_frame_points, (255, 255, 255))
    cv.fillConvexPoly(inpaint_mask, inner_frame_points, (0, 0, 0))

    return inpaint_mask


def gamma_transform(img, gamma):
    '''
     description: 伽马变换
     param {*}
     return {*}
    '''
    # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x/255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # 实现映射用的是Opencv的查表函数
    return cv.LUT(img, gamma_table)


def puttext(img, chinese, pos, color, size=25):
    '''
     description: 支持中文
     param {*}
     return {*}
    '''
    img_PIL = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    font = ImageFont.truetype('NotoSansCJK-Bold.ttc', size)

    draw = ImageDraw.Draw(img_PIL)
    draw.text(pos, chinese, font=font, fill=color)

    return cv.cvtColor(np.asarray(img_PIL), cv.COLOR_RGB2BGR)


def plt_img(img, title=None, size=None, save_name=None, is_show=True):
    '''
     description: plt显示图片
     param {*}
     return {*}
    '''
    plt.figure(num=title, figsize=size)
    plt.title(title)
    if len(img.shape) == 2:
        plt.imshow(img, cmap=plt.cm.gray)
    else:
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    if save_name is not None:
        path = os.path.split(save_name)[0]
        if not os.path.exists(path):
            os.makedirs(path)

        plt.savefig(save_name)

    if is_show:
        plt.show()


def plt_imgs(imgs, shape=(1, 1), titile=None, size=None, sub_titles=None, save_name=None, is_show=True):
    '''
     description: plt批量显示图片
     param {*}
     return {*}
    '''
    fig = plt.figure(num=titile, figsize=size)
    plt.axis('off')

    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    total_plot = shape[0] * shape[1]
    for i in range(total_plot):
        ax = fig.add_subplot(shape[0], shape[1], i+1)
        ax.axis('off')

        if sub_titles is not None and i < len(sub_titles) and sub_titles[i] is not None:
            ax.set_title(sub_titles[i])

        if i < len(imgs) and imgs[i] is not None:
            if len(imgs[i].shape) == 2:
                ax.imshow(imgs[i], cmap='gray')
                # ax.imshow(imgs[i], cmap=plt.cm.gray)
            else:
                ax.imshow(cv.cvtColor(imgs[i], cv.COLOR_BGR2RGB))

    if save_name is not None:
        path = os.path.split(save_name)[0]
        if not os.path.exists(path):
            os.makedirs(path)

        plt.savefig(save_name)

    if is_show:
        plt.show()


def sort_points(points):
    sort_by_y = sorted(points, key=lambda p: p[1])
    p0, p1 = sort_by_y[:2]
    p2, p3 = sort_by_y[2:]

    if p0[0] > p1[0]:
        p0, p1 = p1, p0

    if p3[0] > p2[0]:
        p2, p3 = p3, p2

    return np.array([p0, p1, p2, p3])
