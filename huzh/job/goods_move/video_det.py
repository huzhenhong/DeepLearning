'''
Author: huzhenhong 455879568@qq.com
Date: 2023-12-19 16:52:41
LastEditors: huzhenhong 455879568@qq.com
LastEditTime: 2023-12-19 17:43:30
FilePath: /DeepLearning/huzh/job/goods_move/video_det.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2 as cv
import numpy as np



vd = cv.VideoCapture('/Users/huzh/Documents/algorithm/物品搬移/video/银光/物品搬移/20230920_pick/943工房槽区_20230920151000-20230920173000_8.mp4')


cnt = 0
first = None
diff = None
ma = None

while True:
    cnt += 1
    if not vd.grab():
        break
    if cnt % 12 == 0:
        ret, frame = vd.retrieve()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _, frame = cv.threshold(frame, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            if first is None:
                first = frame
                ma = np.ones_like(frame) * 255
            else:
                diff = cv.absdiff(first, frame)
                
                # an = cv.bitwise_and(first, frame)
                # diff = ma - an
                
                # diff = cv.absdiff(first, frame)
                first = frame
                
                cv.imshow('diff', diff)
                cv.waitKey()
            
        
    
    