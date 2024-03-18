import sys
import cv2 as cv
import numpy as np

sys.path.append('..')

from .get_light import process as cut_light
from .get_color import process as get_color_region
from .judge import process as judge_light


def detect(im):
    h, w, _ = im.shape
    roi = [
        # 0.643,
        # 0.18,
        # 0.665,
        # 0.22,
  0.66125,
      0.1688888888888889,
      0.67875,
      0.20222222222222222
    ]
    # roi = [
    #     0.648,
    #     0.191,
    #     0.661,
    #     0.215,
    # ]
    # roi = [0.642, 0.196, 0.6583, 0.2283]
    roi = [int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)]
    src = im[roi[1] : roi[3], roi[0] : roi[2]]

    # mean = cv.mean(im)
    # print("mean: ", sum(mean))
    # mv = cv.split(im)
    # mean_b = int(mean[0])
    # _, minmax_b, _, _ = cv.minMaxLoc(mv[0])
    # minmax_b = int(minmax_b)

    # mean_g = int(mean[1])
    # _, minmax_g, _, _ = cv.minMaxLoc(mv[1])

    # mean_r = int(mean[2])
    # _, minmax_r, _, _ = cv.minMaxLoc(mv[2])
    # minmax_r = int(minmax_r)

    # print(f'{mean_b} {mean_g} {mean_r} - {minmax_b} {minmax_g} {minmax_r}')

    cv.imshow('src', src)

    light, light_mask = cut_light(src)
    if light is None:
        cv.destroyWindow('light')
    if light_mask is None:
        cv.destroyWindow('light_mask')

    if light is not None and light_mask is not None:
        cv.imshow('light_mask', light_mask)
        cv.imshow('light', light)

        color_mask = get_color_region(light)
        if color_mask is not None:
            intersection = cv.bitwise_and(light_mask, color_mask)
            cv.imshow('intersection', intersection) 
            union = cv.bitwise_or(light_mask, color_mask)
            iou = (
                1.0
                * cv.countNonZero(intersection)
                / (cv.countNonZero(union) + np.finfo(float).eps)
            )
            print('iou is: ', iou)
            if iou > 0.1:
                result = judge_light(src, intersection)
                # result = judge_light(src, light_mask)
                cv.putText(
                    im,
                    result,
                    (roi[0], roi[1]),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
            else:
                print('iou too small')
                cv.putText(
                    im,
                    'LOST',
                    (roi[0], roi[1]),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
        else:
            # 尝试到先前的 mask 里面去找
            print('color_mask is None')
            cv.putText(
                im,
                'LOST',
                (roi[0], roi[1]),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
    else:
        print('light_mask is None')
        cv.putText(
            im,
            'LOST',
            (roi[0], roi[1]),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

    cv.rectangle(im, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 255), 2)
    # cv.rectangle(im, (int(roi[0] * w), int(roi[1] * h)), (int(roi[2] * w), int(roi[3] * h)), (0, 255, 255), 2)
    cv.namedWindow('im', cv.WINDOW_GUI_NORMAL)
    cv.resizeWindow('im', 800, 600)
    
    cv.imshow('im', im)
    cv.waitKey()

    return im
