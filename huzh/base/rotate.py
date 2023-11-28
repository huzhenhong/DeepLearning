# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : rotate image by selected two points
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2020-10-11 12:58:58
 LastEditors  : huzhenhong
 LastEditTime : 2021-03-25 17:11:43
 FilePath     : \\python\\base_library\\rotate.py
 Copyright (C) 2020 huzhenhong. All rights reserved.
'''

import cv2.cv2 as cv
import threading
from pynput import keyboard, mouse
from base_library import cv_function as cvf


class Rotate():
    def __init__(self):
        # 键盘监听
        self.listen_thread = None
        self.mouse = None

        # 旋转图片
        self.rotated = None
        self.start_point = None
        self.end_point = None

    def execute(self, img):
        # 创建键盘监听线程
        self.listen_thread = threading.Thread(target=self.__listen_mouse_and_keyboard)
        self.listen_thread.setDaemon(True)
        self.listen_thread.start()

        self.__rotate(img)

        return self.rotated

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

    def __rotate(self, img):
        cvf.show('rotate', img)
        cv.setMouseCallback('rotate', self.__rotate_by_mouse, img)

        cvf.waitKey()
        cv.destroyAllWindows()

    def __rotate_by_mouse(self, event, x, y, flags, img):
        show_img = img.copy()

        if self.start_point is not None:
            self.__draw_cross(show_img, self.start_point[0], self.start_point[1], (0, 0, 255), thickness=5)

        if self.end_point is not None:
            self.__draw_cross(show_img, self.end_point[0], self.end_point[1], (0, 0, 255), thickness=5)

        if cv.EVENT_LBUTTONDOWN == event:
            self.__draw_cross(show_img, x, y, (0, 0, 255), thickness=5)

        elif cv.EVENT_LBUTTONUP == event:
            # 已经选过一次了，重来
            if self.start_point is not None and self.end_point is not None:
                self.start_point = (x, y)
                self.end_point = None

            # 至少有一个为空
            else:
                if self.start_point is None:
                    self.start_point = (x, y)   # 第一次也进到这里
                else:
                    self.end_point = (x, y)
        elif cv.EVENT_MOUSEMOVE == event:
            self.__draw_cross(show_img, x, y, (0, 255, 0), thickness=5)

        elif cv.EVENT_RBUTTONDOWN == event:
            if self.start_point is not None and self.end_point is not None:
                rotate_angle = cvf.calc_x_angle(self.start_point, self.end_point)
                cv.line(show_img, self.start_point, self.end_point, ((0, 255, 0)), thickness=3, lineType=cv.LINE_AA)
                cv.putText(
                            show_img,
                            'angle: ' + str(rotate_angle),
                            (x + 10, y - 10),
                            cv.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            thickness=2)

                cvf.show('rotate', show_img)

                self.rotated = cvf.rotate(img, rotate_angle)
                cvf.show('rotated', self.rotated)

    def __draw_cross(self, img, x, y, color, thickness=1, is_show=True):
        cv.line(img, (x-10, y), (x+10, y), color, thickness=thickness)
        cv.line(img, (x, y - 10), (x, y + 10), color, thickness=thickness)

        if is_show is True:
            if y - 50 > 0 and y + 50 > 0 and x - 50 > 0 and x + 50 > 0:
                draw_new_point = img[y-50: y+50, x-50: x+50]
                cvf.show('draw point', cvf.resize(draw_new_point, 300))
