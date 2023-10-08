# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  :
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2023-09-13 19:37:19
 LastEditors  : huzhenhong
 LastEditTime : 2023-09-20 16:56:56
 FilePath     : \\DeepLearning\\huzh\\dataset\\convert\\pick_from_coco_to_yolo.py
 Copyright (C) 2023 huzhenhong. All rights reserved.
'''
import os
import shutil
import argparse
import cv2 as cv
from tqdm import tqdm
from pycocotools.coco import COCO


def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
    if xmin >= xmax or ymax <= ymin:
        return 0

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G的面积
    a2 = s1 + s2 - a1
    iou = a1 / s2  # iou = a1/ (s1 + s2 - a1)
    # iou = a1 / a2  # iou = a1/ (s1 + s2 - a1)
    return iou


def clip_coords(boxes, shape):
    boxes[0] = max(0, boxes[0])
    boxes[1] = max(0, boxes[1])
    boxes[2] = min(shape[1], boxes[2])
    boxes[3] = min(shape[0], boxes[3])


def pick_one_label(one_img_anns, label, area, wh_ratio):
    one_label_bbox = []
    for ann in one_img_anns:
        if ann['area'] > area and label == ann['category_id']:
            bbox = list(map(int, ann['bbox']))  # bbox:[x,y,w,h]
            if 1.0 * bbox[2] / bbox[3] > wh_ratio:
                continue
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[0] + bbox[2]
            y2 = bbox[1] + bbox[3]
            one_label_bbox.append([x1, y1, x2, y2])
    return one_label_bbox


def main(opt):
    assert os.path.exists(
        opt.label_path
    ), "label file:{} does not exists".format(opt.label_path)

    assert os.path.exists(opt.img_path), "image path:{} does not exists".format(
        opt.img_path
    )

    if os.path.exists(opt.save_path):
        shutil.rmtree(opt.save_path)
    os.makedirs(os.path.join(opt.save_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(opt.save_path, 'labels'), exist_ok=True)

    coco = COCO(opt.label_path)
    label_2_name = {}
    for cat in coco.dataset['categories']:
        label_2_name[cat['id']] = cat['name']

    all_img_ids = coco.getImgIds()

    file_cnt = 0

    for img_id in tqdm(all_img_ids):
        im_info = coco.loadImgs(img_id)[0]
        sv = im_info['file_name'].split('/')
        patch = sv[2] if opt.is_obj365_2020 else ""
        filename = sv[-1][:-4]
        # filename = im_info['file_name'][10:][:-4]
        # filename = im_info['file_name'][:-4]

        im_path = os.path.join(opt.img_path, patch, filename + '.jpg')
        if not os.path.exists(im_path):
            print("image not exist: ", im_path)
            continue

        category_ids = coco.getAnnIds(imgIds=im_info['id'], iscrowd=None)
        one_img_anns = coco.loadAnns(category_ids)

        person_bboxes = pick_one_label(one_img_anns, 1, 40000, 1.0)
        shoe_bboxes = pick_one_label(one_img_anns, 2, 400, 5.0)

        person_2_shoes = {}
        for shoe in shoe_bboxes:
            for i, person in enumerate(person_bboxes):
                if cal_iou(person, shoe) > 0.95:
                    if i not in person_2_shoes.keys():
                        person_2_shoes[i] = []
                    person_2_shoes[i].append(shoe)

        if len(person_2_shoes) == 0:
            continue

        if not os.path.exists(os.path.join(opt.save_path, 'images', patch)):
            os.makedirs(
                os.path.join(opt.save_path, 'images', patch), exist_ok=True
            )
            os.makedirs(
                os.path.join(opt.save_path, 'labels', patch), exist_ok=True
            )

        im = cv.imread(im_path)
        im_height, im_width = im.shape[:2]

        # new_shoe_yolo_list = []
        for i, shoes in person_2_shoes.items():
            person = person_bboxes[i]
            clip_coords(person, im.shape)

            w = person[2] - person[0]
            h = person[3] - person[1]
            w_offset = int(w * 0.03 + 0.5)  # rather more
            h_offset = int(h * 0.03 + 0.5)
            left_padding = w_offset if person[0] - w_offset > 0 else 0
            top_padding = h_offset if person[1] - h_offset > 0 else 0
            right_padding = (
                w_offset
                if person[2] + w_offset < im_width
                else im_width - person[2]
            )

            bottom_padding = (
                h_offset
                if person[3] + h_offset < im_height
                else im_height - person[3]
            )

            x1 = person[0] - left_padding
            y1 = person[1] - top_padding
            x2 = person[2] + right_padding
            y2 = person[3] + bottom_padding
            p_xyxy = [x1, y1, x2, y2]

            crop = im[p_xyxy[1] : p_xyxy[3], p_xyxy[0] : p_xyxy[2]]
            final_height, final_width = crop.shape[:2]
            cv.imwrite(
                os.path.join(
                    opt.save_path, 'images', patch, f'{filename}_{i}.jpg'
                ),
                crop,
            )

            new_shoe_yolo_list = []
            for shoe in shoes:
                clip_coords(shoe, im.shape)
                s_xyxy = [
                    shoe[0] - person[0] + left_padding,
                    shoe[1] - person[1] + top_padding,
                    shoe[2] - person[0] + left_padding,
                    shoe[3] - person[1] + top_padding,
                ]
                # clip_coords(s_xyxy, crop.shape)

                s_yolo = (
                    (s_xyxy[0] + s_xyxy[2]) * 0.5 / final_width,
                    (s_xyxy[1] + s_xyxy[3]) * 0.5 / final_height,
                    (s_xyxy[2] - s_xyxy[0]) / final_width,
                    (s_xyxy[3] - s_xyxy[1]) / final_height,
                )

                new_shoe_yolo_list.append(s_yolo)

            with open(
                os.path.join(
                    opt.save_path, 'labels', patch, f'{filename}_{i}.txt'
                ),
                'w',
            ) as f_writer:
                file_cnt += 1
                # print(file_cnt, f'{filename}_{i}.txt')
                for xyxy in new_shoe_yolo_list:
                    line = '1 {:.5f} {:.5f} {:.5f} {:.5f}\n'.format(
                        xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                    )
                    # line = '0' + ' '.join(xyxy) + '\n'
                    f_writer.write(line)
    with open(os.path.join(opt.save_path, 'classes.txt'), 'w') as f:
        f.write("shoe")

    print("total file: ", file_cnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l',
        '--label_path',
        type=str,
        default='/data/lyjhome/data/obj365-2020/train/zhiyuan_objv2_train.json',
        # default='/data/lyjhome/data/obj365-2020/val/zhiyuan_objv2_val.json',
        help='cocso json label path',
    )
    parser.add_argument(
        '-i',
        '--img_path',
        type=str,
        default='/data/lyjhome/data/obj365-2020/images/train',
        # default='/data/lyjhome/data/obj365-2020/images/val',
        help='image path',
    )
    parser.add_argument(
        '-d',
        '--save_path',
        type=str,
        default='pick_person_shoe_train',
        # default='pick_person_shoe_val',
        help='save path',
    )
    parser.add_argument(
        '--src_id',
        nargs='+',
        default=1,
        # required=True,
        help='<Required> Set flag',
    )
    parser.add_argument(
        '--dst_id',
        nargs='+',
        default=1,
        # required=True,
        help='<Required> Set flag',
    )
    parser.add_argument(
        '--is_obj265_2020',
        type=bool,
        default=False,
        help='is_obj265_2020',
    )

    opt = parser.parse_args()
    print('\n')
    print(opt)
    print('\n')

    main(opt)
