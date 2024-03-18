import os
import sys
import cv2 as cv

# import numpy as np
sys.path.append('..')
from base.utils import get_video_info, get_specify_files

from light_switch.detect import detect

# from pick_color.color_inrange import detect
# from goods_move.post_process import process as detect


def parse_argument():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input_path',
        type=str,
        help='image or image folder or video or video input path',
        # required=True,
        default='/Users/huzh/Documents/project/中车/指示灯开关检测/20231114/局放控制台灯20231109/灯亮',
    )
    parser.add_argument(
        '-o',
        '--output_path',
        type=str,
        help='image or image folder or video or video output path',
        # required=True,
        default='output',
    )
    # parser.add_argument(
    #     '-e',
    #     '--ext',
    #     type=list,
    #     nargs='*',
    #     help='input file extension',
    #     # required=True,
    #     default=['.jpg', '.mp4'],
    # )
    return parser.parse_args()


def process_images(images_path, save_path):
    for im_path in images_path:
        im = cv.imread(im_path)
        if im is None:
            print('read image [{}] failed', im_path)
        else:
            draw_result = detect(im)
            if draw_result is not None:
                cv.imwrite(
                    os.path.join(save_path, os.path.basename(im_path)),
                    draw_result,
                )


def process_videos(videos_path, save_path, frame_step):
    for vd_path in videos_path:
        vd_reader = cv.VideoCapture(vd_path)
        if not vd_reader.isOpened():
            print('open video [{}] failed.', vd_path)
            continue

        im_width, im_height, count, fps, channels = get_video_info(vd_reader)
        print(
            'video [{}] w x h: {} x {} count: {} fps: {} channels: {}',
            vd_path,
            im_width,
            im_height,
            count,
            fps,
            channels,
        )

        vd_writer = cv.VideoWriter(
            os.path.join(save_path, os.path.basename(vd_path)),
            cv.VideoWriter_fourcc('H', '2', '6', '4'),
            fps,
            (im_width, im_height),
        )
        if not vd_writer.isOpened():
            print('open file args.save_path failed')
            continue

        cnt = 0
        while cnt < count:
            cnt += 1
            if not vd_reader.grab():
                continue
            elif cnt % frame_step == 0:
                ret, im = vd_reader.retrieve()
                if ret:
                    detect(im)
                    vd_writer.write(im)

        vd_writer.release()


def main(args):
    if not os.path.exists(args.input_path):
        print(f'[{args.input_path}] do not exist.')
        exit(-1)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    image_suffixes = ['.jpg', 'jpeg', '.bmp', '.png']
    video_suffixes = ['.mp4', '.avi', '.flv', '.h264', '.ts']

    image_path = get_specify_files(args.input_path, image_suffixes)
    video_path = get_specify_files(args.input_path, video_suffixes)

    if len(image_path) == 0:
        print(f"no image in [{args.input_path}]")
        # exit(-1)
    process_images(image_path, args.output_path)

    if len(video_path) == 0:
        print(f"no video in [{args.input_path}]")
        # exit(-1)
    process_videos(video_path, args.output_path, 5)


if __name__ == '__main__':
    main(parse_argument())
