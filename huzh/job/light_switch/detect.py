
import sys
import cv2 as cv
import numpy as np
sys.path.append('..')

from .get_light import process as cut_light
from .get_color import process as get_color_region
from .judge import process as judge_light


def detect(im):
    h, w, _ = im.shape
    roi = [0.642, 0.196, 0.658, 0.228]
    roi = [int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)]
    src = im[roi[1] : roi[3], roi[0] : roi[2]]

    light, light_mask = cut_light(src)

    if light is not None and light_mask is not None:
        color_mask = get_color_region(light)
        if color_mask is not None:
            intersection = cv.bitwise_and(light_mask, color_mask)
            union = cv.bitwise_or(light_mask, color_mask)
            iou = 1.0 * cv.countNonZero(intersection) / (cv.countNonZero(union) + np.finfo(float).eps)
            print('iou is: ', iou)
            if iou > 0.5:
                result = judge_light(src, light_mask)
                cv.putText(im, result, (roi[0], roi[1]), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                print('iou too small')
                cv.putText(im, 'LOST', (roi[0], roi[1]), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            print('color_mask is None')
            cv.putText(im, 'LOST', (roi[0], roi[1]), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        print('light_mask is None')
        cv.putText(im, 'LOST', (roi[0], roi[1]), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv.imshow('im', im)
    # cv.waitKey()
    # return im

