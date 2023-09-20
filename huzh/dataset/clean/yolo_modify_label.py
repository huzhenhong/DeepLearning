# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  :
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2023-09-20 14:19:32
 LastEditors  : huzhenhong
 LastEditTime : 2023-09-20 14:34:15
 FilePath     : \\DeepLearning\\huzh\\dataset\\clean\\yolo_modify_label.py
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

    src_labels_path = args.src_path
    out_labels_path = args.out_path
    os.makedirs(out_labels_path)

    for labelfile in tqdm(os.listdir(src_labels_path)):
        labelfile_path = os.path.join(src_labels_path, labelfile)
        if labelfile == 'classes.txt' or not labelfile.endswith('.txt'):
            print('error labelfile: ', labelfile_path)
            continue

        if os.stat(labelfile_path).st_size == 0:
            print('empty labelfile: ', labelfile_path)
            continue

        with open(labelfile_path, 'r') as f_reader:
            lines = f_reader.readlines()
            with open(os.path.join(out_labels_path, labelfile), 'w') as f_writer:
                for line in lines:
                    line = '1' + line[1:]
                    f_writer.write(line)


if __name__ == '__main__':
    main(parse_argument())
