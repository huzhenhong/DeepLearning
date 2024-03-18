import cv2 as cv
import numpy as np


def parse_argument():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v1',
        '--video1_path',
        type=str,
        help='video path',
        # required=True,
        default='/Users/huzh/Documents/algorithm/物品搬移/video/good_move_test0.mp4',
    )
    # parser.add_argument(
    #     '-v2',
    #     '--video2_path',
    #     type=str,
    #     help='video2 path',
    #     # required=True,
    #     default='/Users/huzh/Documents/github/SAM/segment-anything/samexporter/debug_with_previrous.mp4',
    # )
    parser.add_argument(
        '-s',
        '--save_path',
        type=str,
        help='save path',
        # required=True,
        default='/Users/huzh/Documents/algorithm/物品搬移/video/good_move_test0_long.mp4',
    )
    return parser.parse_args()


def get_video_info(vd_reader):
    im_width = int(vd_reader.get(cv.CAP_PROP_FRAME_WIDTH))
    im_height = int(vd_reader.get(cv.CAP_PROP_FRAME_HEIGHT))
    count = int(vd_reader.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(vd_reader.get(cv.CAP_PROP_FPS))
    channels = int(vd_reader.get(cv.CAP_PROP_VIDEO_TOTAL_CHANNELS))
    return (im_width, im_height, count, fps, channels)


def main(args):
    vd1_reader = cv.VideoCapture(args.video1_path)
    # vd2_reader = cv.VideoCapture(args.video2_path)
    vd1_prop = get_video_info(vd1_reader)
    # vd2_prop = get_video_info(vd2_reader)
    if not vd1_reader.isOpened():
        print('open args.video1_path failed')
        exit(-1)

    # if not vd2_reader.isOpened():
    #     print('open file args.video2_path failed')
    #     exit(-1)

    # if vd1_prop != vd2_prop:
    #     print('prop do not match')
    #     # exit(-1)

    vd_writer = cv.VideoWriter(
        args.save_path,
        cv.VideoWriter_fourcc('H', '2', '6', '4'),
        vd1_prop[3],
        (vd1_prop[0], vd1_prop[1]),
    )
    if not vd_writer.isOpened():
        print('open file args.save_path failed')
        exit(-1)

    i = 0
    last_frame = None
    while True:
        # i += 1
        ret1, im1 = vd1_reader.read()
        # ret2, im2 = vd2_reader.read()
        if ret1:
            # im = np.vstack((im1, im2))
            vd_writer.write(im1)
            last_frame = im1
        else:
            print('read frame failed')
            # exit(-1)
            # continue
            # break
            total_append_frame_cnt = 180 * vd1_prop[3]
            print("total_append_frame_cnt: ", total_append_frame_cnt)
            while i < total_append_frame_cnt:
                i += 1
                vd_writer.write(last_frame)
            break

    vd_writer.release()


if __name__ == '__main__':
    main(parse_argument())
