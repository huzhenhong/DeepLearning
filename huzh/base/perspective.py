# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : perspective image by selected four points
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2020-10-11 12:58:58
 LastEditors  : huzhenhong
 LastEditTime : 2021-03-17 11:45:31
 FilePath     : \\Job\\base_library\\perspective.py
 Copyright (C) 2020 huzhenhong. All rights reserved.
'''

import cv2.cv2 as cv
import numpy as np
import threading
from pynput import keyboard, mouse
import base_library.cv_function as cvf


class Perspective():
    def __init__(self):
        # 键盘监听
        self.listen_thread = None
        self.mouse = None

        # 透视变换挑选的点
        self.chosed_points_list = []
        self.perspectived = None

    def execute(self, img):
        # 创建键盘监听线程
        self.listen_thread = threading.Thread(target=self.__listen_mouse_and_keyboard)
        self.listen_thread.setDaemon(True)
        self.listen_thread.start()

        # 手动选择四个点进行透视变换
        return self.__perspective_transform(img)

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

    def __perspective_transform(self, img):
        cvf.cv_show('chose four points', img)
        cv.setMouseCallback('chose four points', self.__chose_four_points_by_mouse, img)

        cvf.cv_waitKey()
        cv.destroyAllWindows()

        return self.perspectived

    def __chose_four_points_by_mouse(self, event, x, y, flags, img):
        draw_all_points = img.copy()
        for point in self.chosed_points_list:
            self.__draw_cross(draw_all_points, point[0], point[1], (0, 0, 255), thickness=5)
        cv.imshow('chose four points', draw_all_points)

        if cv.EVENT_LBUTTONUP == event:
            if len(self.chosed_points_list) >= 4:
                self.chosed_points_list.pop(0)

            self.chosed_points_list.append((x, y))
            self.__draw_cross(draw_all_points, x, y, (0, 0, 255), thickness=5)

        elif cv.EVENT_MOUSEMOVE == event:
            self.__draw_cross(draw_all_points, x, y, (0, 255, 0))

        elif cv.EVENT_RBUTTONDOWN == event:
            if len(self.chosed_points_list) == 4:
                src_points = self.__sort_points(self.chosed_points_list)

                self.perspectived = cvf.perspective_transform(img, src_points.astype(np.float32))
                cvf.cv_show('perspectived', self.perspectived)

    def __draw_cross(self, img, x, y, color, thickness=1, is_show=True):
        cv.line(img, (x-10, y), (x+10, y), color, thickness=thickness)
        cv.line(img, (x, y - 10), (x, y + 10), color, thickness=thickness)

        if is_show is True:
            if y - 50 > 0 and y + 50 > 0 and x - 50 > 0 and x + 50 > 0:
                draw_new_point = img[y-50: y+50, x-50: x+50]
                cv.imshow('draw point', cvf.cv_resize(draw_new_point, 300))

    def __sort_points(self, points):
        sort_by_y = sorted(points, key=lambda p: p[1])
        p0, p1 = sort_by_y[:2]
        p2, p3 = sort_by_y[2:]

        if p0[0] > p1[0]:
            p0, p1 = p1, p0

        if p3[0] > p2[0]:
            p2, p3 = p3, p2

        return np.array([p0, p1, p2, p3])


obj = Perspective()
img = cv.imread('002_gate_status_detection/data/img/1.png')
perspective = obj.execute(cv.imread('002_gate_status_detection/data/img/1.png'))
cv.imwrite('perspective1.jpg', perspective)
