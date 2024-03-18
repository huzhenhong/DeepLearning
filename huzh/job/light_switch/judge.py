import cv2 as cv


def process(im, mask):
    light = cv.bitwise_and(im, im, mask=mask)
    cv.imshow('light_final', light)
    mean = cv.mean(im, mask=mask)
    print("mean: ", sum(mean))
    # minmax_b = cv.minMaxLoc(im, mask=mask)
    mv = cv.split(im)

    mean_b = int(mean[0])
    _, minmax_b, _, _ = cv.minMaxLoc(mv[0], mask=mask)
    minmax_b = int(minmax_b)

    mean_g = int(mean[1])
    _, minmax_g, _, _ = cv.minMaxLoc(mv[1], mask=mask)

    mean_r = int(mean[2])
    _, minmax_r, _, _ = cv.minMaxLoc(mv[2], mask=mask)
    minmax_r = int(minmax_r)

    print(f'{mean_b} {mean_g} {mean_r} - {minmax_b} {minmax_g} {minmax_r}')

    # if mean_r > 190 and minmax_r > 250:
    if int(sum(mean)) > 300:
        return 'ON'
        # cv.putText(im, 'on', (roi[0], roi[1]), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        # cv.putText(im, 'off', (roi[0], roi[1]), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return 'OFF'
