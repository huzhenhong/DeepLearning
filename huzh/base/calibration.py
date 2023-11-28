# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 标定多边形，按ESC退出
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2021-03-17 11:46:38
 LastEditors  : huzhenhong
 LastEditTime : 2021-03-25 17:39:30
 FilePath     : \\python\\base_library\\calibration.py
 Copyright    : All rights reserved.
'''

import cv2 as cv
import numpy as np
import threading
from pynput import keyboard, mouse
import base_library.cv_function as cvf


class Calibration():
    def __init__(self, ploy):
        self.ploy = ploy
        self.listen_thread = None
        self.mouse = None
        self.chosed_points_list = []
        self.roi_mask = None

    def execute(self, img):
        # 创建键盘监听线程，用来微调点位置
        self.listen_thread = threading.Thread(target=self.__listen_mouse_and_keyboard)
        self.listen_thread.setDaemon(True)
        self.listen_thread.start()

        # 手动选择标定点
        cvf.show('chose points', img)
        cv.setMouseCallback('chose points', self.__chose_points_by_mouse, img)

        cvf.waitKey()
        cv.destroyAllWindows()

        return self.chosed_points_list, self.roi_mask

    def __listen_mouse_and_keyboard(self):
        self.mouse = mouse.Controller()

        with keyboard.Listener(on_press=self.__listen_keyboard) as listener:
            listener.join()

    def __listen_keyboard(self, e):
        if e == keyboard.Key.up:
            self.mouse.move(0, -1)
        elif e == keyboard.Key.down:
            self.mouse.move(0, 1)
        elif e == keyboard.Key.left:
            self.mouse.move(-1, 0)
        elif e == keyboard.Key.right:
            self.mouse.move(1, 0)
        elif e == keyboard.Key.esc:
            return False

    def __chose_points_by_mouse(self, event, x, y, flags, img):
        draw_all_points = img.copy()
        for point in self.chosed_points_list:
            self.__draw_cross(draw_all_points, point[0], point[1], (0, 0, 255), thickness=5)
        cvf.show('chose points', draw_all_points)

        if cv.EVENT_LBUTTONUP == event:
            if len(self.chosed_points_list) >= self.ploy:
                self.chosed_points_list.pop(0)

            self.chosed_points_list.append([x, y])
            self.__draw_cross(draw_all_points, x, y, (0, 0, 255), thickness=5)
            cvf.show('chose points', self.__draw_ploy(draw_all_points, self.chosed_points_list))

        elif cv.EVENT_MOUSEMOVE == event:
            self.__draw_cross(draw_all_points, x, y, (0, 255, 0))
            cvf.show('chose points', self.__draw_ploy(draw_all_points, self.chosed_points_list))

        elif cv.EVENT_RBUTTONDOWN == event:
            # 抠图
            self.roi_mask = np.zeros_like(draw_all_points)
            self.roi_mask = cv.fillPoly(self.roi_mask, [np.array(self.chosed_points_list)], (255, 255, 255))
            roi = cv.bitwise_and(img, self.roi_mask)
            cvf.show('roi', roi)

    def __draw_cross(self, img, x, y, color, thickness=1, is_show=True):
        cv.line(img, (x-10, y), (x+10, y), color, thickness=thickness)
        cv.line(img, (x, y - 10), (x, y + 10), color, thickness=thickness)

        if is_show is True:
            if y - 50 > 0 and y + 50 > 0 and x - 50 > 0 and x + 50 > 0:
                draw_new_point = img[y-50: y+50, x-50: x+50]
                cvf.show('draw point', cvf.resize(draw_new_point, 300))

    def __draw_ploy(self, img, points):
        points = np.array(points, np.int32)
        if len(points) >= 2:
            cv.polylines(img, [points], True, (0, 0, 255), 2)

        return img
