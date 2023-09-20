# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  :
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2023-09-18 17:52:46
 LastEditors  : huzhenhong
 LastEditTime : 2023-09-18 18:00:25
 FilePath     : \\DeepLearning\\others\\dataset_clean\\yolo_clean_label.py
 Copyright (C) 2023 huzhenhong. All rights reserved.
'''
import os
import shutil
from tqdm import tqdm


def parse_argument():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--src_path',
        type=str,
        help='dataset path for clean',
        required=True,
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help='dataset path for write',
        required=True,
    )
    return parser.parse_args()


def main(args):
    if os.path.exists(args.out_path):
        shutil.rmtree(args.out_path)

    src_labels_path = os.path.join(args.src_path, 'labels')
    # src_labels_path = args.src_path
    src_images_path = os.path.join(args.src_path, 'images')
    out_labels_path = os.path.join(args.out_path, 'labels')
    # out_labels_path = args.out_pat
    out_images_path = os.path.join(args.out_path, 'images')
    os.makedirs(out_labels_path)
    os.makedirs(out_images_path)

    for labelfile in tqdm(os.listdir(src_labels_path)):
        labelfile_path = os.path.join(src_labels_path, labelfile)
        if labelfile == 'classes.txt' or not labelfile.endswith('.txt'):
            print('error labelfile: ', labelfile_path)
            continue

        if os.stat(labelfile_path).st_size == 0:
            print('empty labelfile: ', labelfile_path)
            continue

        im_path = os.path.join(src_images_path, labelfile.replace('.txt', '.jpg'))
        if not os.path.exists(im_path):
            print('image not exiest: ', im_path)
            continue

        with open(labelfile_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if int(line[0]) != 0:
                    shutil.move(
                        labelfile_path, os.path.join(out_labels_path, labelfile)
                    )
                    shutil.move(
                        im_path, os.path.join(out_images_path, labelfile.replace('.txt', '.jpg'))
                    )
                    break


if __name__ == '__main__':
    main(parse_argument())
