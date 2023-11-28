# !usr/bin/env python
# -*- coding:utf-8 -*-
'''
 Description  : 
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2021-07-05 09:38:35
 LastEditors  : huzhenhong
 LastEditTime : 2021-11-11 16:37:14
 FilePath     : \\Tools\\make_dataset_seat_belt.py
 Copyright (C) 2021 huzhenhong. All rights reserved.
'''

import os
from pathlib import Path
import shutil
from tqdm import tqdm
import random

date = '20211111'
root = Path('E://dataset/seat_belt/20211001/')
src_label_path = root / date / 'labels-filter'
src_img_path = root / date / 'images-filter'
dst_label_path = root / 'seat_belt' / 'labels'
dst_img_path = root / 'seat_belt' / 'images'

txt_list = [
    path for path in src_label_path.rglob('*.txt')
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

    img_path = src_img_path / Path(path.name).with_suffix('.jpg')
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

    img_path = src_img_path / Path(path.name).with_suffix('.jpg')
    dst_path = dst_img_path / 'val' / Path(path.name).with_suffix('.jpg')
    # os.symlink(src_path, dst_path)
    if not dst_path.parent.exists():
        dst_path.parent.mkdir(parents=True)
    shutil.copy(img_path, dst_path)
