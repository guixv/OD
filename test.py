import json

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from safetensors.torch import load_model, save_model

from utils.utils import Evaluation
import argparse
import numpy as np
import os

from modules.convnext import convnext_tiny
from modules.VGG import vgg
from modules.Resnet import res
from modules.densenet import dense
from modules.ViT import vit16
from modules.swinmodel import swin_base_patch4_window7_224
from modules.swinTransformer import swin_transformer
from modules.Res2Net import res2
from modules.vit_model import vit_base_patch16_224 
from modules.mobilenet_v4 import MNV4ConvMedium,MNV4HybridMedium
from modules.timm_MNV4 import mobilenetv4_conv_medium,mobilenetv4_conv_blur_medium
from modules.pvtv2 import pvt_v2_b3
import sys


def main():
    parse = argparse.ArgumentParser(description="classification")
    parse.add_argument("--batch_size", type=int, default=128)
    parse.add_argument("--lr", type=int, default=0.001)
    parse.add_argument("--lrf", type=int, default=0.0001)
    parse.add_argument("--input_size", type=int, default=256)
    parse.add_argument("--weight", type=str, default="./output/MNV4_blur.safetensors")
    # parse.add_argument("--log_eval", type=str, default="output/ViT/log_val.txt")
    # parse.add_argument("--log_list", type=str, default="output/ViT/log_list.txt")
    parse.add_argument("--train_path", type=str, default="/data/work_folder/data/train_data/ILSVRC2012/train")
    parse.add_argument("--val_path", type=str, default="/data/work_folder/data/train_data/ILSVRC2012/val")
    parse.add_argument("--class_num", type=int, default=1000)
    parse.add_argument("--output_path", type=str, default="output/safe/")

    opt = parse.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    
    test(opt, device)

    return 0

def test(opt, device):
    input_size = opt.input_size
    class_num = opt.class_num
    batch_size = opt.batch_size
    weight = opt.weight

    

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}

    
    model = mobilenetv4_conv_blur_medium(num_classes=1000, features_only=False,pretrained= False)
    # model = MNV4ConvMedium(num_classes=class_num)
    # model = nn.DataParallel(model)
    model.to(device)
    test_path = opt.val_path
    
    if weight != "":
        ###################safetensor
        load_model(model, weight)


        ###################pth
        # assert os.path.exists(weight), "weights file: '{}' not exist.".format(weight)
        # weights_dict = torch.load(weight, map_location=device)

        # # #多卡转单卡
        # # from collections import OrderedDict
        # # new_state_dict = OrderedDict()
        # # for k, v in weights_dict.items():
        # #     name = k[7:] # 去掉 `module.`
        # #     new_state_dict[name] = v
        # # # 加载参数
        # # print(model.load_state_dict(new_state_dict))
        # # 删除有关分类类别的权重
        # # for k in list(weights_dict.keys()):
        # #     if "head" in k:
        # #         del weights_dict[k]

        # #单卡转多卡
        # # from collections import OrderedDict
        # # new_state_dict = OrderedDict()
        # # for k, v in weights_dict.items():
        # #     name = 'module.'+k # 去掉 `module.`
        # #     new_state_dict[name] = v
        # # # 加载参数
        # # print(model.load_state_dict(new_state_dict))

        # #正常
        # print(model.load_state_dict(weights_dict, strict=False))

    test_data = torchvision.datasets.ImageFolder(root=test_path, transform=data_transform["test"])
    testdata = DataLoader(dataset=test_data, batch_size=batch_size // 2, shuffle=True,
                          num_workers=8)  # 将训练数据以每次n张图片的形式抽出进行测试

    test_size = len(test_data)  # 测试集的长度
    print("using {} images for validation.".format( test_size))  # 用于打印总的训练集数量和验证集数量
    criterion = nn.CrossEntropyLoss().to(device)
    
    test_eval = test_epoch(0, model, testdata, criterion, device)


def test_epoch(epoch, model, testdata, criterion, device):
    model.eval()
    true = []
    pre = []
    losses = 0

    with torch.no_grad():
        for image, label in tqdm(testdata):
            image, label = Variable(image.float()).to(device), Variable(label).to(device)

            output = model(image)
            loss = criterion(output, label)
            pre_ = torch.argmax(output, 1)
            true.extend(label.tolist())
            pre.extend(pre_.tolist())
            losses += loss.item()
        Loss = losses / len(testdata)

    Accuracy, Precision, Recall, F1Score = Evaluation(pre, true)
    print("Test  Epoch({}): Acc:{} Prec:{} Recall:{} F1-score:{} Loss:{}".format(epoch, Accuracy, Precision, Recall,
                                                                                 F1Score, Loss))

    return Accuracy, Precision, Recall, F1Score, Loss


if __name__ == '__main__':
    main()
