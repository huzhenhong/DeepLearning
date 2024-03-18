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

import os
import sys
import cv2 as cv
import numpy as np
from pathlib import Path

sys.path.append('..')
from base.utils import get_video_info, get_specify_files


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

        contours, _ = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        sorted_contours = sorted(
            contours, key=lambda c: cv.contourArea(c), reverse=True
        )
        draw_img = cv.drawContours(
            self.__im.copy(), sorted_contours, 0, (255, 0, 255), 1
        )

        if len(sorted_contours) > 1:
            draw_img = cv.drawContours(
                draw_img, sorted_contours[1:], -1, (255, 255, 0), 1
            )

        mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        cv.imshow('in_range', np.hstack((draw_img, mask)))

        # fill_img = np.zeros_like(self.__im)
        # fill_img += np.array((255, 0, 255), dtype=np.uint8)
        # fill_img = cv.bitwise_and(fill_img, fill_img, mask=mask)

        # draw_img = cv.addWeighted(self.__im, 0.5, fill_img, 0.5, 0)
        # cv.imshow('in_range', np.hstack((self.__im, draw_img)))


def detect(im):
    h, w, _ = im.shape
    # roi = [0.642, 0.196, 0.658, 0.228]
    roi = [
        0.6474999315261892,
        0.18888885912483325,
        0.6649999304771477,
        0.2199999685781995,
    ]

    roi = [
        int(roi[0] * w),
        int(roi[1] * h),
        int(roi[2] * w),
        int(roi[3] * h),
    ]
    src = im[roi[1] : roi[3], roi[0] : roi[2]]
    ColorInRange(src, 'yellow')
    cv.waitKey()


# def parse_argument():
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '-i',
#         '--input_path',
#         type=str,
#         help='image or image folder or video or video input path',
#         # required=True,
#         default='/Users/huzh/Documents/project/中车/指示灯开关检测/20231114/局放控制台灯20231109/灯亮',
#     )
#     parser.add_argument(
#         '-o',
#         '--output_path',
#         type=str,
#         help='image or image folder or video or video output path',
#         # required=True,
#         default='output',
#     )
#     # parser.add_argument(
#     #     '-e',
#     #     '--ext',
#     #     type=list,
#     #     nargs='*',
#     #     help='input file extension',
#     #     # required=True,
#     #     default=['.jpg', '.mp4'],
#     # )
#     return parser.parse_args()


# def process_images(images_path, save_path):
#     for im_path in images_path:
#         im = cv.imread(im_path)
#         if im is None:
#             print('read image [{}] failed', im_path)
#         else:
#             draw_result = detect(im)
#             if draw_result is not None:
#                 cv.imwrite(
#                     os.path.join(save_path, os.path.basename(im_path)),
#                     draw_result,
#                 )


# def process_videos(videos_path, save_path, frame_step):
#     for vd_path in videos_path:
#         vd_reader = cv.VideoCapture(vd_path)
#         if not vd_reader.isOpened():
#             print('open video [{}] failed.', vd_path)
#             continue

#         im_width, im_height, count, fps, channels = get_video_info(vd_reader)
#         print(
#             'video [{}] w x h: {} x {} count: {} fps: {} channels: {}',
#             vd_path,
#             im_width,
#             im_height,
#             count,
#             fps,
#             channels,
#         )

#         vd_writer = cv.VideoWriter(
#             os.path.join(save_path, os.path.basename(vd_path)),
#             cv.VideoWriter_fourcc('H', '2', '6', '4'),
#             fps,
#             (im_width, im_height),
#         )
#         if not vd_writer.isOpened():
#             print('open file args.save_path failed')
#             continue

#         cnt = 0
#         while cnt < count:
#             cnt += 1
#             if not vd_reader.grab():
#                 continue
#             elif cnt % frame_step == 0:
#                 ret, im = vd_reader.retrieve()
#                 if ret:
#                     detect(im)
#                     vd_writer.write(im)

#         vd_writer.release()


# def main(args):
#     if not os.path.exists(args.input_path):
#         print(f'[{args.input_path}] do not exist.')
#         exit(-1)

#     if not os.path.exists(args.output_path):
#         os.makedirs(args.output_path)

#     image_suffixes = ['.jpg', 'jpeg', '.bmp', '.png']
#     video_suffixes = ['.mp4', '.avi', '.flv', '.h264', '.ts']

#     image_path = get_specify_files(args.input_path, image_suffixes)
#     video_path = get_specify_files(args.input_path, video_suffixes)

#     if len(image_path) == 0:
#         print(f"no image in [{args.input_path}]")
#         # exit(-1)
#     process_images(image_path, args.output_path)

#     if len(video_path) == 0:
#         print(f"no video in [{args.input_path}]")
#         # exit(-1)
#     process_videos(video_path, args.output_path, 5)


# if __name__ == '__main__':
#     main(parse_argument())
