# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  :
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2023-09-13 17:24:55
 LastEditors  : huzhenhong
 LastEditTime : 2023-09-14 11:40:36
 FilePath     : \\DeepLearning\\others\\label_convert\\show_coco_dataset.py
 Copyright (C) 2023 huzhenhong. All rights reserved.
'''
import os
import json
import shutil
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO


def draw_bbox(draw_img, one_img_show_info, is_bgr=False):
    """_summary_

    Args:
        one_img_show_info (_type_): draw_img, category_id, category_name, x1, y1, x2, y2
    """

    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

    hex = (
        '0000FF',
        'FF3838',
        'FF9D97',
        'FF701F',
        'FFB21D',
        'CFD231',
        '48F90A',
        '92CC17',
        '3DDB86',
        '1A9334',
        '00D4BB',
        '2C99A8',
        '00C2FF',
        '344593',
        '6473FF',
        '0018EC',
        '8438FF',
        '520085',
        'CB38FF',
        'FF95C8',
        'FF37C7',
    )

    palette = [hex2rgb('#' + i) for i in hex]

    for one_ann_show_info in one_img_show_info:
        category_id, category_name, x1, y1, x2, y2 = one_ann_show_info

        color = palette[int(category_id) % len(palette)]
        color = (color[2], color[1], color[0]) if is_bgr else color

        cv.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)
        cv.putText(
            draw_img,
            category_name,
            (x1, y1),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return draw_img


def show_statistics(statistics_info):
    # 绘制每种类别个数柱状图
    plt.bar(
        range(len(statistics_info)),
        statistics_info.values(),
        align='center',
    )

    # 将横坐标0,1,2,3,4替换为相应的类别名称
    plt.xticks(
        range(len(statistics_info)), statistics_info.keys(), rotation=90
    )

    # 在柱状图上添加数值标签
    for index, (i, v) in enumerate(statistics_info.items()):
        plt.text(x=index, y=v, s=str(v), ha='center')

    # 设置x坐标
    plt.xlabel('category name')

    # 设置y坐标
    plt.ylabel('number of objects')

    # 设置柱状图的标题
    plt.title('category distribution')

    plt.savefig("category_distribution.png")
    plt.show()


def main(opt):
    assert os.path.exists(
        opt.label_path
    ), "label file:{} does not exists".format(opt.label_path)

    assert os.path.exists(opt.img_path), "image path:{} does not exists".format(
        opt.img_path
    )

    if os.path.exists(opt.save_path):
        shutil.rmtree(opt.save_path)
    os.makedirs(opt.save_path)

    statistics_info = {}

    coco = COCO(opt.label_path)
    label_2_name = {}
    for cat in coco.dataset['categories']:
        label_2_name[cat['id']] = cat['name']

    all_img_ids = coco.getImgIds()

    for img_id in tqdm(all_img_ids):
        im_info = coco.loadImgs(img_id)[0]
        filename = im_info['file_name']
        category_ids = coco.getAnnIds(imgIds=im_info['id'], iscrowd=None)
        one_img_anns = coco.loadAnns(category_ids)

        one_img_show_info = []
        draw_img = cv.imread(os.path.join(opt.img_path, filename))

        for ann in one_img_anns:

            category_id = ann['category_id']
            category_name = label_2_name[category_id]

            bbox = list(map(int, ann['bbox']))  # bbox:[x,y,w,h]
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[0] + bbox[2]
            y2 = bbox[1] + bbox[3]

            one_img_show_info.append(
                [category_id, category_name, x1, y1, x2, y2]
            )

            if opt.statistics_label:
                if category_name not in statistics_info.keys():
                    statistics_info[category_name] = 0
                statistics_info[category_name] += 1

        if opt.show_img or opt.write_img:
            img = draw_bbox(draw_img, one_img_show_info)

            if opt.show_img:
                cv.imshow(filename, img)
                cv.waitKey()
                cv.destroyAllWindows()

            if opt.write_img:
                cv.imwrite(os.path.join(opt.save_path, filename), img)

    if opt.statistics_label:
        show_statistics(statistics_info)

    print('statistics_info:', statistics_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l',
        '--label_path',
        type=str,
        # default='/Users/huzh/Documents/数据集/Objects365-2019-zip/Annotations/val/val.json',
        default='/Users/huzh/Documents/数据集/Objects365-2019-zip/Annotations/val/picked_val.json',
        help='coco json label path',
    )
    parser.add_argument(
        '-i',
        '--img_path',
        type=str,
        default='/Users/huzh/Documents/数据集/Objects365-2019/images/val',
        help='image path',
    )
    parser.add_argument(
        '-d',
        '--save_path',
        type=str,
        default='debug',
        help='image save path',
    )
    parser.add_argument(
        '-s',
        '--show_img',
        type=bool,
        default=True,
        help='min object area',
    )
    parser.add_argument(
        '-w',
        '--write_img',
        type=bool,
        default=True,
        help='min object area',
    )
    parser.add_argument(
        '-st',
        '--statistics_label',
        type=bool,
        default=True,
        help='statistics every label',
    )

    opt = parser.parse_args()
    print('\n')
    print(opt)
    print('\n')

    main(opt)
