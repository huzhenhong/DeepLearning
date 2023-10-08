# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Description  :
 Version      : 1.0
 Author       : huzhenhong
 Date         : 2023-09-12 11:47:20
 LastEditors  : huzhenhong
 LastEditTime : 2023-09-20 15:18:55
 FilePath     : \\tools\\pick_dataset_from_yolo_dataset.py
 Copyright (C) 2023 huzhenhong. All rights reserved.
'''
import os
import re
import shutil
import cv2.cv2 as cv
from tqdm import tqdm


def parse_arguments():
    import argparse

    # import ast
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path of yolo dataset.")
    parser.add_argument("--dst", type=str, required=True, help="Path of dst.")
    # parser.add_argument(
    #     "--topk", type=int, default=1, help="Return topk results."
    # )
    parser.add_argument(
        '--src_id', nargs='+', required=True, help='<Required> Set flag'
    )
    parser.add_argument(
        '--dst_id', nargs='+', required=True, help='<Required> Set flag'
    )
    parser.add_argument(
        "--type",
        type=str,
        default='train',
        help="Type of dataset, support 'train' or 'val'.",
    )
    parser.add_argument(
        "--draw_bbox", action='store_true', help="Wether to draw bbox."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    cnt = 0

    if os.path.exists(args.dst):
        # raise Exception(f'path [args.dst] exist!')
        shutil.rmtree(args.dst)

    src_images_root = os.path.join(args.src, 'images', args.type)
    src_labels_root = os.path.join(args.src, 'labels', args.type)
    dst_images_root = os.path.join(args.dst, 'images', args.type)
    dst_labels_root = os.path.join(args.dst, 'labels', args.type)

    for patch in tqdm(os.listdir(src_labels_root)):
        path_dir = os.path.join(src_labels_root, patch)
        for labelfile in os.listdir(path_dir):
            img = cv.imread(
                os.path.join(
                    src_images_root,
                    patch,
                    labelfile.replace('.txt', '.jpg'),
                )
            )
            h, w = img.shape[:2]
            if h < 500 or w < 500:
                continue

            with open(os.path.join(path_dir, labelfile)) as f_reader:
                lines = f_reader.readlines()
                pick_lines = []
                for line in lines:
                    class_id = line.split()[0]
                    # class_id = int(line.split()[0])
                    if class_id in args.src_id:
                        sv = line.split(' ')
                        x1 = int(
                            (float(sv[1]) - float(sv[3]) / 2) * w
                        )  # x_center - width/2
                        y1 = int(
                            (float(sv[2]) - float(sv[4]) / 2) * h
                        )  # y_center - height/2
                        x2 = int(
                            (float(sv[1]) + float(sv[3]) / 2) * w
                        )  # x_center + width/2
                        y2 = int(
                            (float(sv[2]) + float(sv[4]) / 2) * h
                        )  # y_center + height/2
                        shoe_w = x2 - x1
                        shoe_h = y2 - y1
                        if (
                            shoe_w < 20
                            or shoe_h < 20
                            or shoe_w > 100
                            or shoe_h > 100
                        ):
                            continue

                        new_class_id = str(
                            args.dst_id[args.src_id.index(class_id)]
                        )
                        new_class_id = class_id
                        # have_veh = 1
                        new_line = re.sub(r'^\d+', new_class_id, line)
                        pick_lines.append(new_line)

                if len(pick_lines) > 0:
                    cnt += 1
                    # continue

                    output_label_path = os.path.join(dst_labels_root, patch)
                    if not os.path.exists(output_label_path):
                        os.makedirs(output_label_path)

                    with open(
                        os.path.join(output_label_path, labelfile), "w"
                    ) as f_writer:
                        f_writer.writelines(pick_lines)

                    # copy image
                    output_img_path = os.path.join(dst_images_root, patch)
                    if not os.path.exists(output_img_path):
                        os.makedirs(output_img_path)

                    image_file = labelfile.replace('.txt', '.jpg')
                    shutil.copyfile(
                        os.path.join(src_images_root, patch, image_file),
                        os.path.join(dst_images_root, patch, image_file),
                    )

                    if args.draw_bbox:
                        print('bbox num: ', len(pick_lines))
                        draw_img = cv.imread(
                            os.path.join(dst_images_root, patch, image_file)
                        )
                        for line in pick_lines:
                            sv = line.split(' ')
                            h, w = draw_img.shape[:2]
                            x1 = int(
                                (float(sv[1]) - float(sv[3]) / 2) * w
                            )  # x_center - width/2
                            y1 = int(
                                (float(sv[2]) - float(sv[4]) / 2) * h
                            )  # y_center - height/2
                            x2 = int(
                                (float(sv[1]) + float(sv[3]) / 2) * w
                            )  # x_center + width/2
                            y2 = int(
                                (float(sv[2]) + float(sv[4]) / 2) * h
                            )  # y_center + height/2
                            cv.rectangle(
                                draw_img, (x1, y1), (x2, y2), (0, 0, 255), 1
                            )
                            cv.putText(
                                draw_img,
                                sv[0],
                                (x1, y1),
                                cv.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                            )
                        cv.imwrite(
                            os.path.join(dst_images_root, patch, image_file),
                            draw_img,
                        )
        print(cnt)
    print(cnt)
