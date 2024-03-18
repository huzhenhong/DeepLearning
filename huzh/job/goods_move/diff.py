'''
Author: huzhenhong 455879568@qq.com
Date: 2023-12-19 16:04:26
LastEditors: huzhenhong 455879568@qq.com
LastEditTime: 2023-12-22 17:13:16
FilePath: /DeepLearning/huzh/job/goods_move/diff.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2 as cv
import numpy as np


im0 = cv.imread('/Users/huzh/Pictures/Screenshots/yinguang_fensuijian-0002.jpg')
im1 = cv.imread('/Users/huzh/Pictures/Screenshots/yinguang_fensuijian-0004.jpg')
# im0 = cv.imread('/Users/huzh/Documents/gitlab/goodsmovedetector/goods_move_debug/1606工房2_20230920151000-20230920173000_3.mp4/2023-12-19 14_01_13.189773_4552980504/2023-12-19 15_10_02.593043/ref.jpg')
# im1 = cv.imread('/Users/huzh/Documents/gitlab/goodsmovedetector/goods_move_debug/1606工房2_20230920151000-20230920173000_3.mp4/2023-12-19 14_01_13.189773_4552980504/2023-12-19 15_10_02.593043/cur.jpg')


gray0 = cv.cvtColor(im0, cv.COLOR_BGR2GRAY)
_, binary0 = cv.threshold(gray0, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
gray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
_, binary1 = cv.threshold(gray1, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)


diff = cv.absdiff(binary0, binary1)
# diff = cv.absdiff(im0, im1)
cv.imshow('diff', diff)
# cv.waitKey()

# gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
# _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# # _, binary = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)

# cv.imshow('binary', binary)
# # cv.imshow('binary1', binary1)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
open1 = cv.morphologyEx(diff, cv.MORPH_OPEN, kernel, iterations=1)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
# close1 = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=1)
# close2 = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=2)
close3 = cv.morphologyEx(open1, cv.MORPH_CLOSE, kernel, iterations=3)
# close4 = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=4)
# dilate = cv.dilate(binary, (3, 3), 1)
# cv.imshow('dilate', dilate)
cv.imshow('diff', diff)
cv.imshow('open1', open1)
# cv.imshow('close2', close2)
cv.imshow('close3', close3)
# cv.imshow('close4', close4)


cv.waitKey()
