# !usr/bin/env python
# -*- coding:utf-8 -*-
'''
 Description  : 
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2021-07-05 09:38:35
 LastEditors  : huzhenhong
 LastEditTime : 2021-11-17 11:31:00
 FilePath     : \\Tools\\make_cls_dataset.py
 Copyright (C) 2021 huzhenhong. All rights reserved.
'''

import os
from pathlib import Path
import shutil
from tqdm import tqdm
import random

root = Path('E://dataset/ganfeng_work_clothes/20211108/images/')
cls_map = {}

for dir in root.iterdir():
    img_path_list = []
    for img_path in dir.iterdir():
        img_path_list.append(str(img_path))
    cls_map[dir.name] = img_path_list

train_list = []
valid_list = []

for name, img_path_list in cls_map.items():
    # 切分训练集和验证集
    random.shuffle(img_path_list)
    train_size = int(len(img_path_list) * 0.9)

    for i in range(train_size):
        train_list.append(img_path_list[i] + ',' + name + '\n')

    random.shuffle(train_list)

    for i in range(train_size, len(img_path_list)):
        valid_list.append(img_path_list[i] + ',' + name + '\n')

    random.shuffle(valid_list)

with open('trainset_list.txt', mode='w') as f:
    for line in train_list:
        f.write(line)

with open('validset_list.txt', mode='w') as f:
    for line in valid_list:
        f.write(line)
