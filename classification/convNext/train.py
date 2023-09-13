# 导入相应的库
import argparse
import math
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataLoader.dataSet import read_split_data
from dataLoader.dataLoader import My_Dataset
from utils import (
    matplotlib_imshow,
    train_one_epoch,
    evaluate,
    get_params_groups,
    create_lr_scheduler,
)
from models.networks import convnext_tiny


# 主函数
def main(opt):
    # 1.读取一些配置参数，并且输出
    print(opt)
    assert os.path.exists(opt.data_path), "{} dose not exists.".format(
        opt.data_path
    )

    # 创建日志文件
    tb_writer = SummaryWriter()
    # 日志保存路径
    save_dir = tb_writer.log_dir
    # 模型保存路径
    weights_dir = save_dir + "/weights"
    # 如果文件夹不存在，则创建文件夹
    os.makedirs(weights_dir, exist_ok=True)

    # 设备
    device = torch.device(
        'cuda' if torch.cuda.is_available() and opt.use_cuda else "cpu"
    )

    # 2.数据读取
    (
        train_images_path,
        val_images_path,
        train_labels,
        val_labels,
        every_class_num,
    ) = read_split_data(
        data_root=opt.data_path, val_rate=0.2, save_dir=save_dir
    )

    # 3.pytorch框架
    #   3.1 数据加载：dataset,transform,dataloader
    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        ),
    }

    train_dataset = My_Dataset(
        images_path=train_images_path,
        images_class=train_labels,
        transform=data_transform['train'],
    )
    val_dataset = My_Dataset(
        images_path=val_images_path,
        images_class=val_labels,
        transform=data_transform['val'],
    )

    nw = min(
        [
            os.cpu_count(),
            opt.batch_size,
            opt.num_worker if opt.batch_size > 1 else 0,
            8,
        ]
    )
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True,
    )

    #   3.2 网络搭建：model
    classes = len(every_class_num)
    model = convnext_tiny(num_classes=classes)
    if opt.weights != '':
        assert os.path.exists(
            opt.weights
        ), "weights file: '{}' not exist.".format(opt.weights)
        weights_dict = torch.load(opt.weights, map_location="cpu")
        in_channel = model.head.in_features
        model.fc = nn.Linear(in_channel, classes)
        del_keys = ['head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict['model'][k]
        model.load_state_dict(weights_dict, strict=False)

    if opt.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad(False)
            else:
                print("training {}".format(name))
    model = model.to(device)

    #   3.3 优化器，学习率，更新策略,损失函数
    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=5e-2)
    if opt.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            pg, lr=opt.lr, momentum=0.9, weight_decay=5e-5
        )
    elif opt.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(pg, lr=opt.lr, weight_decay=1e-3)
    elif opt.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            pg, lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-2
        )

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - opt.lrf) + opt.lrf  # cosine
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)

    scheduler = create_lr_scheduler(
        optimizer, len(train_loader), opt.epochs, warmup=True, warmup_epochs=1
    )
    loss_function = torch.nn.CrossEntropyLoss()

    #   3.4 模型训练：train
    best_acc = -np.inf
    best_epoch = 0
    for epoch in tqdm(range(opt.epochs)):
        # train
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            loss_function,
            epoch=epoch,
            scheduler=scheduler,
        )

        #  eval
        val_loss, val_acc = evaluate(
            model, val_loader, device, loss_function, epoch
        )
        tags = [
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "learning_rate",
            "images",
        ]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        batch_images = next(iter(train_loader))[0]
        tb_writer.add_images(tags[5], batch_images, epoch)

        #   3.6 模型保存：save
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch + 1
        model_path = weights_dir + "/model_{}.pth".format(epoch)
        torch.save(model.state_dict(), model_path)
        if is_best:
            shutil.copy(model_path, weights_dir + "/best_model.pth")
    tb_writer.close()


# 程序入口
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path', type=str, default='./data', help='The data path'
    )
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-worker', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lrf', type=float, default=0.01)

    # https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
    parser.add_argument(
        '--weights',
        type=str,
        default='convnext_tiny_1k_224_ema.pth',
        help='initial weights path',
    )  #
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument(
        '--optimizer', type=str, default='adamw'
    )  # sgd,adam,adamw

    args = parser.parse_args()

    main(opt=args)
