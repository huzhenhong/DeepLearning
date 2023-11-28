import cv2 as cv
import numpy as np


def process(im):
    # Setting parameter values
    t_lower = 50  # Lower Threshold
    t_upper = 150  # Upper threshold

    # Applying the Canny Edge filter
    edge = cv.Canny(im, t_lower, t_upper)

    edge_3c = cv.cvtColor(edge, cv.COLOR_GRAY2BGR)

    cv.imshow('edge', np.vstack((im, edge_3c)))

    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    # equ = cv.equalizeHist(gray)
    # cv.imshow('equ', np.hstack((gray, equ)))

    # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # equ = clahe.apply(gray)

    _, binary = cv.threshold(
        gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV
    )
    # _, binary = cv.threshold(
    #     gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV
    # )

    binary_3c = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    save_img = np.hstack((im, binary_3c))

    cv.imshow('save_img', save_img)
    cv.waitKey()

    return save_img
