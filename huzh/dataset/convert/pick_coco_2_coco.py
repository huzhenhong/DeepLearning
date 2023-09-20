# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  :
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2023-09-13 16:01:28
 LastEditors  : huzhenhong
 LastEditTime : 2023-09-13 19:33:52
 FilePath     : \\DeepLearning\\others\\label_convert\\pick_coco_2_coco.py
 Copyright (C) 2023 huzhenhong. All rights reserved.
'''
import os
import json
import argparse
from tqdm import tqdm
from pycocotools.coco import COCO


def filter_coco(opt):
    assert os.path.exists(
        opt.src_path
    ), "src json path:{} does not exists".format(opt.src_path)

    if os.path.exists(opt.save_path):
        os.remove(opt.save_path)

    dst_coco = dict()
    dst_coco['type'] = 'instances'
    dst_coco['images'] = []
    dst_coco['annotations'] = []
    dst_coco['categories'] = []

    src_coco = COCO(opt.src_path)
    all_img_ids = src_coco.getImgIds()
    classesIds = src_coco.getCatIds()
    classesIds.sort(reverse=False)

    dst_coco['categories'] = [
        cat
        for cat in src_coco.dataset['categories']
        if cat['id'] in opt.src_ids
    ]

    for img_id in tqdm(all_img_ids):
        im_info = src_coco.loadImgs(img_id)[0]
        width = im_info['width']
        height = im_info['height']

        if width < opt.img_width or height < opt.img_height:
            continue

        one_img_ann_id = src_coco.getAnnIds(imgIds=im_info['id'], iscrowd=None)
        one_img_anns = src_coco.loadAnns(one_img_ann_id)

        is_have_object = False
        for ann in one_img_anns:
            if ann['category_id'] in opt.src_ids and ann['area'] > opt.object_area:
                dst_coco['annotations'].append(ann)
                is_have_object = True

        if is_have_object:
            dst_coco['images'].append(im_info)

    print('total pick imgs:', len(dst_coco['images']))
    print('total pick objects:', len(dst_coco['annotations']))

    json.dump(dst_coco, open(opt.save_path, 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--src_path',
        type=str,
        default='/Users/huzh/Documents/数据集/Objects365-2019-zip/Annotations/val/val.json',
        help='src json path',
    )
    parser.add_argument(
        '-d',
        '--save_path',
        type=str,
        default='/Users/huzh/Documents/数据集/Objects365-2019-zip/Annotations/val/picked_val.json',
        help='dst json path',
    )
    parser.add_argument(
        '-area',
        '--object_area',
        type=int,
        default='500',
        help='min object area',
    )
    parser.add_argument(
        '-width',
        '--img_width',
        type=int,
        default='500',
        help='min image width',
    )
    parser.add_argument(
        '-height',
        '--img_height',
        type=int,
        default='500',
        help='min image height',
    )
    parser.add_argument(
        '-i',
        '--src_ids',
        type=list,
        nargs='*',
        default=[1, 2],
        help='picked id',
    )
    parser.add_argument(
        '-o',
        '--dst_ids',
        type=list,
        nargs='*',
        default=[1, 0],
        help='min image height',
    )
    opt = parser.parse_args()
    print('\n')
    print(opt)
    print('\n')

    filter_coco(opt)
