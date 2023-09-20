# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  :
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2023-09-18 10:41:03
 LastEditors  : huzhenhong
 LastEditTime : 2023-09-18 11:10:52
 FilePath     : \\DeepLearning\\others\\dataset_clean\\make_detect_dataset_from_multiple_dir.py
 Copyright (C) 2023 huzhenhong. All rights reserved.
'''


import os
import shutil
import random
from tqdm import tqdm


def parse_argument():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--src_path',
        type=str,
        help='dataset source path',
        required=True,
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help='make dataset out path',
        required=True,
    )
    parser.add_argument(
        '-tr',
        '--train_ratio',
        type=float,
        help='train ratio',
        default=0.9,
    )

    try:
        return parser.parse_args()
    except Exception as e:
        print(e)
        parser.print_help()


def main(args):
    if os.path.exists(args.out_path):
        shutil.rmtree(args.out_path)

    src_labels_path = os.path.join(args.src_path, 'labels')
    src_images_path = os.path.join(args.src_path, 'images')
    out_train_image_path = os.path.join(args.out_path, 'images', 'train')
    out_train_label_path = os.path.join(args.out_path, 'labels', 'train')
    out_val_image_path = os.path.join(args.out_path, 'images', 'val')
    out_val_label_path = os.path.join(args.out_path, 'labels', 'val')
    os.makedirs(out_train_image_path)
    os.makedirs(out_train_label_path)
    os.makedirs(out_val_image_path)
    os.makedirs(out_val_label_path)

    labelfile_list = []
    for labelfile in tqdm(os.listdir(src_labels_path)):
        labelfile_path = os.path.join(src_labels_path, labelfile)
        if labelfile == 'classes.txt' or not labelfile.endswith('.txt'):
            print('error labelfile: ', labelfile_path)
            continue

        if os.stat(labelfile_path).st_size == 0:
            print('empty labelfile: ', labelfile_path)
            continue

        im_path = os.path.join(
            src_images_path, labelfile.replace('.txt', '.jpg')
        )
        if not os.path.exists(im_path):
            print('image not exiest: ', im_path)
            continue

        labelfile_list.append(labelfile)

    random.shuffle(labelfile_list)
    train_num = int(len(labelfile_list) * args.train_ratio)
    train_label_list = labelfile_list[:train_num]
    val_label_list = labelfile_list[train_num:]

    for labelfile in tqdm(train_label_list, 'trian'):
        shutil.copyfile(
            os.path.join(src_labels_path, labelfile),
            os.path.join(out_train_label_path, labelfile),
        )
        shutil.copyfile(
            os.path.join(src_images_path, labelfile.replace('.txt', '.jpg')),
            os.path.join(
                out_train_image_path, labelfile.replace('.txt', '.jpg')
            ),
        )

    for labelfile in tqdm(val_label_list, 'val'):
        shutil.copyfile(
            os.path.join(src_labels_path, labelfile),
            os.path.join(out_val_label_path, labelfile),
        )
        shutil.copyfile(
            os.path.join(src_images_path, labelfile.replace('.txt', '.jpg')),
            os.path.join(
                out_val_image_path, labelfile.replace('.txt', '.jpg')
            ),
        )


if __name__ == '__main__':
    main(parse_argument())

