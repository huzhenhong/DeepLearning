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
    parser.add_argument(
        '-m',
        '--mode',
        type=bool,
        help='is save into sub folder',
        # required=True,
        default=False,
    )
    return parser.parse_args()


def process_videos(vd_path, save_path, video_name, frame_step):
    # for vd_path in videos_path:
    vd_reader = cv.VideoCapture(vd_path)
    if not vd_reader.isOpened():
        print('open video [{}] failed.', vd_path)
        return

    im_width, im_height, count, fps, channels = get_video_info(vd_reader)
    print(
        f'video [{vd_path}] w x h: {im_width} x {im_height} count: {count} fps: {fps} channels: {channels}'
    )

    cnt = 0
    while cnt < count:
        cnt += 1
        if not vd_reader.grab():
            continue
        elif cnt % frame_step == 0:
            ret, im = vd_reader.retrieve()
            if ret:
                im = im[:][0: -52]
                cv.imwrite(os.path.join(save_path, f'{video_name}_{cnt}.jpg'), im)
                print('im_path: ', os.path.join(save_path, f'{video_name}_{cnt}.jpg'))


def main(args):
    if not os.path.exists(args.input_path):
        print(f'[{args.input_path}] do not exist.')
        exit(-1)

    output_path = os.path.join(args.output_path, os.path.split(args.input_path)[-1])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # image_suffixes = ['.jpg', 'jpeg', '.bmp', '.png']
    video_suffixes = ['.mp4', '.avi', '.flv', '.h264', '.ts']

    # image_path = get_specify_files(args.input_path, image_suffixes)
    video_path = get_specify_files(args.input_path, video_suffixes)

    if len(video_path) == 0:
        print(f"no video in [{args.input_path}]")
        exit(-1)
    
    for vd_path in video_path:
        if args.mode is True:
            video_name = os.path.splitext(os.path.basename(vd_path))[0]
            im_save_path = os.path.join(output_path, os.path.dirname(vd_path))
            if not os.path.exists(im_save_path):
                os.makedirs(im_save_path, True)
                os.chmod(im_save_path, 0o777)
            process_videos(vd_path, 
                           im_save_path, 
                           video_name,
                           args.frame_step)
        else:
            im_save_path = output_path
            rel_path = os.path.relpath(vd_path, args.input_path)
            rel_path = os.path.splitext(rel_path)[0]
            rel_path_split = os.path.split(rel_path)
            rel_path_split = [s for s in rel_path_split if s != '']
            save_video_name = "_".join(rel_path_split)
            process_videos(vd_path, 
                           im_save_path, 
                           save_video_name,
                           args.frame_step)


if __name__ == '__main__':
    main(parse_argument())
