# !usr/bin/env python
# -*- coding:utf-8 -*-
'''
 Description  : 
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2021-07-05 09:38:35
 LastEditors  : huzhenhong
 LastEditTime : 2021-12-08 13:54:24
 FilePath     : \\Tools\\make_detect_dataset_multiple_dir.py
 Copyright (C) 2021 huzhenhong. All rights reserved.
'''

import os
from pathlib import Path
import shutil
from tqdm import tqdm
import random

name = 'negative'
root = Path('E://dataset/ganfeng_work_clothes/防护面罩/标注/valid_data/')
images_dir_list = root / (name + '_images')
labels_dir_list = root / (name + '_labels')

dst_label_path = root / 'mixed_dataset' / name / 'labels'
dst_img_path = root / 'mixed_dataset' / name / 'images'

txt_list = [
    path for path in labels_dir_list.rglob('*.txt')
    if path.name != 'classes.txt' and os.stat(str(path)).st_size != 0
]

random.shuffle(txt_list)

train_num = int(len(txt_list) * 0.9)

for path in tqdm(txt_list[:train_num], 'train'):
    dst_path = dst_label_path / 'train' / Path(path.name)
    # os.symlink(path, label_path)
    if not dst_path.parent.exists():
        dst_path.parent.mkdir(parents=True)
    shutil.copy(path, dst_path)

    img_path = Path(str(Path(path).parent).replace('labels', 'images')) / Path(
        path.name).with_suffix('.jpg')
    dst_path = dst_img_path / 'train' / Path(path.name).with_suffix('.jpg')
    # os.symlink(src_path, dst_path)
    if not dst_path.parent.exists():
        dst_path.parent.mkdir(parents=True)
    shutil.copy(img_path, dst_path)

for path in tqdm(txt_list[train_num:], 'val'):
    dst_path = dst_label_path / 'val' / Path(path.name)
    # os.symlink(path, label_path)
    if not dst_path.parent.exists():
        dst_path.parent.mkdir(parents=True)
    shutil.copy(path, dst_path)

    img_path = Path(str(Path(path).parent).replace('labels', 'images')) / Path(
        path.name).with_suffix('.jpg')
    dst_path = dst_img_path / 'val' / Path(path.name).with_suffix('.jpg')
    # os.symlink(src_path, dst_path)
    if not dst_path.parent.exists():
        dst_path.parent.mkdir(parents=True)
    shutil.copy(img_path, dst_path)
