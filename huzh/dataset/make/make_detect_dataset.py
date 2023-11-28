# !usr/bin/env python
# -*- coding:utf-8 -*-
'''
 Description  :
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2021-07-05 09:38:35
 LastEditors  : huzhenhong
 LastEditTime : 2021-12-10 19:11:30
 FilePath     : \\Tools\\make_detect_dataset.py
 Copyright (C) 2021 huzhenhong. All rights reserved.
'''

import os
from pathlib import Path
import shutil
from tqdm import tqdm
import random

root = Path('/Users/huzh/Documents/dataset/劳保鞋/pick_from_objects365_2019_val_clean/')
images_dir_list = [
    'images',
]
labels_dir_list = [
    'labels',
]

dst_label_path = root / 'xx_dataset' / 'labels'
dst_img_path = root / 'xx_dataset' / 'images'


txt_list = []
for label_dir in labels_dir_list:
    txt_list.extend(
        path
        for path in (root / label_dir).rglob('*.txt')
        if path.name != 'classes.txt' and os.stat(str(path)).st_size != 0
    )

random.shuffle(txt_list)

train_num = int(len(txt_list) * 0.9)

for path in tqdm(txt_list[:train_num], 'train'):
    dst_path = dst_label_path / 'train' / Path(path.name)
    # os.symlink(path, label_path)
    if not dst_path.parent.exists():
        dst_path.parent.mkdir(parents=True)
    shutil.copy(path, dst_path)

    img_path = Path(str(Path(path).parent).replace('labels', 'images')) / Path(
        path.name
    ).with_suffix('.jpg')
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
        path.name
    ).with_suffix('.jpg')
    dst_path = dst_img_path / 'val' / Path(path.name).with_suffix('.jpg')
    # os.symlink(src_path, dst_path)
    if not dst_path.parent.exists():
        dst_path.parent.mkdir(parents=True)
    shutil.copy(img_path, dst_path)
