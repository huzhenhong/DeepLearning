import os
import sys
import cv2 as cv

# import numpy as np
sys.path.append('..')
from base.utils import get_video_info, get_specify_files


# from pick_color.color_inrange import detect
# from goods_move.post_process import process as detect


def parse_argument():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input_path',
        type=str,
        help='video or video folder path',
        required=True,
        default=None,
    )
    parser.add_argument(
        '-o',
        '--output_path',
        type=str,
        help='image output path',
        # required=True,
        default='output',
    )
    parser.add_argument(
        '-f',
        '--frame_step',
        type=int,
        help='pick frame step',
        # required=True,
        default=25,
    )
    return parser.parse_args()


def process_videos(vd_path, save_path, frame_step):
    # for vd_path in videos_path:
    vd_reader = cv.VideoCapture(vd_path)
    if not vd_reader.isOpened():
        print('open video [{}] failed.', vd_path)
        return

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

    cnt = 0
    while cnt < count:
        cnt += 1
        if not vd_reader.grab():
            continue
        elif cnt % frame_step == 0:
            ret, im = vd_reader.retrieve()
            if ret:
                cv.imwrite(os.path.join(save_path, f'{cnt}.jpg'), im)
                print('im_path: ', os.path.join(save_path, f'{cnt}.jpg'))


def main(args):
    if not os.path.exists(args.input_path):
        print(f'[{args.input_path}] do not exist.')
        exit(-1)

    # if not os.path.exists(args.output_path):
    #     os.makedirs(args.output_path)

    # image_suffixes = ['.jpg', 'jpeg', '.bmp', '.png']
    video_suffixes = ['.mp4', '.avi', '.flv', '.h264', '.ts']

    # image_path = get_specify_files(args.input_path, image_suffixes)
    video_path = get_specify_files(args.input_path, video_suffixes)

    if len(video_path) == 0:
        print(f"no video in [{args.input_path}]")
        exit(-1)
    
    for vd_path in video_path:
        im_save_path = os.path.join(args.output_path, 
                                    os.path.splitext(os.path.basename(vd_path))[0])
        if not os.path.exists(im_save_path):
            os.makedirs(im_save_path, True)
            os.chmod(im_save_path, 0o777)

        process_videos(vd_path, im_save_path, args.frame_step)


if __name__ == '__main__':
    main(parse_argument())
