# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  : 
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2021-08-02 17:50:34
 LastEditors  : huzhenhong
 LastEditTime : 2021-08-04 15:08:27
 FilePath     : \\python\\base_library\\realtime_stream.py
 Copyright (C) 2021 huzhenhong. All rights reserved.
'''

import cv2 as cv
import numpy as np


def run_on_camera(func, camera_index=0, winname='camera'):
    cap = cv.VideoCapture(camera_index)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    print('fps', fps)

    if cap.isOpened():
        cv.namedWindow(winname, cv.WINDOW_NORMAL)
        cv.resizeWindow(winname, 640, 480)
    else:
        print(f'open camera {camera_index} failed.')
        return

    while True:
        try:
            ret_val, frame = cap.read()
        except Exception as e:
            print('exception', repr(e))

        if ret_val is True:
            func(frame)
        else:
            print('read frame failed.')

        k = cv.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv.destroyWindow(winname)


def run_on_video(func, video_file_path, stride=1, winname='video'):
    cap = cv.VideoCapture(video_file_path)
    frame_total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    print('frame_total', frame_total)
    print('fps', fps)

    if cap.isOpened():
        cv.namedWindow(winname, cv.WINDOW_NORMAL)
        cv.resizeWindow(winname, 640, 480)
    else:
        print(f'open video {video_file_path} failed.')
        return

    is_run = True

    def progress_callback(index):
        global is_run
        if index == frame_total:
            is_run = False

    cv.createTrackbar('frame_index', winname, 0, frame_total, progress_callback)

    while is_run:
        frame_index = cv.getTrackbarPos('frame_index', 'frame')
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)

        try:
            ret_val, frame = cap.read()
        except Exception as e:
            print('exception', repr(e))

        if ret_val is True:
            func(frame)
        else:
            print('read frame failed.')

        if frame_index + stride > frame_total:
            break
        else:
            cv.setTrackbarPos('frame_index', 'frame', frame_index + stride)

        k = cv.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv.destroyWindow(winname)
