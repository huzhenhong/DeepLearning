import json

import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import cv2
import sys
import argparse
import os
from PIL import Image
import numpy as np
import glob
import time
import shutil
from tqdm import tqdm

from torchvision.models import resnet50
from models import get_RepVGG_func_by_name, func_dict

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


@torch.no_grad()
def run(
    arch='RepVGG-A0',  # 网络名字
    weights='best_model.pth',  # 模型路径
    mode='deploy',  # train or deploy
    source='./data/test',  # 测试数据路径，可以是文件夹，可以是单张图片
    use_cuda=True,  # 是否使用cuda
    view_img=False,  # 是否可视化测试图片
    save_txt=True,  # 是否将结果保存到txt
    project='runs/result',  # 结果输出路径
    class_indices='class_indices.json',  # json文件，存放类别和索引的关系。
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    )

    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if save_txt:
        if os.path.exists(project):
            shutil.rmtree(project)
        os.makedirs(project)
        f = open(project + "/result.txt", 'w')

    # read class_indict
    json_path = class_indices
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(
        json_path
    )

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    num_classes = len(class_indict)

    # load model
    assert os.path.exists(weights), "model path: {} does not exists".format(
        weights
    )
    assert arch in func_dict
    regvgg_build_func = get_RepVGG_func_by_name(arch)
    model = regvgg_build_func(deploy=mode == 'deploy', num_classes=num_classes)

    model.load_state_dict(
        torch.load(weights, map_location=device)["state_dict"], strict=True
    )
    model.eval().to(device)

    # run once
    y = model(torch.rand(1, 3, 224, 224).to(device))

    # load img
    assert os.path.exists(source), "data source: {} does not exists".format(
        source
    )
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, '*.*')))
    elif os.path.isfile(source):
        # img = Image.open(source)
        # if img.mode != 'RGB':
        #     img = img.convert('RGB')
        files = [source]
    else:
        raise Exception(f'ERROR: {source} does not exist')

    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
    images = tqdm(images)
    image_list = []
    for img_path in images:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # [N,C,H,W]
        # 单张图片预测
        img_tensor = data_transform(img)[None, :]

        # 多张打包成patch进行预测
        # image_list.append(data_transform(img))
        # image_tensor = torch.stack(image_list,dim=0)

        t1 = time_sync()
        pred = model(img_tensor.to(device))
        t2 = time_sync()
        pred_class = torch.max(pred, dim=1)[1]
        c = pred_class.cpu().numpy().item()
        prob = torch.squeeze(torch.softmax(pred, dim=1)).cpu().numpy()[int(c)]
        # print("name:{}\tclass: {}\tprob: {:.3}\tinference time: {:.5f}s Done.".format(img_path.split(os.sep)[-1],c, prob, (t2 - t1)))
        print(
            "class: {}\tprob: {:.3}\tinference time: {:.5f}s Done.".format(
                class_indict[str(c)], prob, (t2 - t1)
            )
        )

        # 可视化图片
        if view_img:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            cv2.imshow("image", img)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img, class_indict[str(c)], (14, 14), font, 1, (0, 0, 255), 3
            )
            cv2.waitKey()
        if save_txt:
            file_name = img_path.split(os.sep)[-1]
            # f.write("{} {}\n".format(file_name, c))
            f.write("{} {}\n".format(file_name, class_indict[str(c)]))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='RepVGG-A0')
    parser.add_argument(
        '--weights', type=str, default='model_best.pth', help='the model path'
    )
    parser.add_argument(
        '--mode', type=str, default='train', help='train or deploy'
    )
    parser.add_argument(
        '--source', type=str, default='./data/test', help='test data path'
    )
    parser.add_argument('--use-cuda', type=bool, default=True)
    parser.add_argument('--view-img', type=bool, default=False)
    parser.add_argument('-s', '--save-txt', type=bool, default=True)
    parser.add_argument(
        '--project', type=str, default='runs/result', help='output path'
    )
    parser.add_argument(
        '--class-indices',
        type=str,
        default='class_indices.json',
        help='when train,the file will generate',
    )
    opt = parser.parse_args()
    run(**vars(opt))
