import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils import Evaluation
import argparse
import numpy as np
import os
import sys


class VGG(nn.Module):
    def __init__(self, features, num_classes, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        self.fc = nn.Linear(4096, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg(model_name="vgg13", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]
    model = VGG(make_layers(cfg, batch_norm=False), **kwargs)
    return model


def train_epoch(epoch, model, traindata, criterion, optimizer, device):
    model.train()
    true = []
    pre = []
    losses = 0
    for image, label in tqdm(traindata):
        image, label = Variable(image.float()).to(device), Variable(label).to(device)

        optimizer.zero_grad()  # 初始化梯度值
        output = model(image)

        loss = criterion(output, label)
        loss.backward()  # 反向求解梯度
        losses += loss.item()
        optimizer.step()  # 更新参数

        pre_ = np.argmax(output.detach().cpu().numpy(), axis=1)
        true.extend(label.tolist())
        pre.extend(pre_.tolist())
    Loss = losses / len(traindata)
    Accuracy, Precision, Recall, F1Score = Evaluation(pre, true)
    print("Train Epoch({}): Acc:{} Prec:{} Recall:{} F1-score:{} Loss:{}".format(epoch, Accuracy, Precision, Recall,
                                                                                 F1Score, Loss))

    return Accuracy, Precision, Recall, F1Score, Loss


def test_epoch(epoch, model, testdata, criterion, device):
    model.eval()
    true = []
    pre = []
    losses = 0

    for image, label in tqdm(testdata):
        image, label = Variable(image.float()).to(device), Variable(label).to(device)

        output = model(image)
        pre_ = np.argmax(output.detach().cpu().numpy(), axis=1)
        true.extend(label.tolist())
        pre.extend(pre_.tolist())
        loss = criterion(output, label)
        losses += loss.item()
    Loss = losses / len(testdata)

    Accuracy, Precision, Recall, F1Score = Evaluation(pre, true)
    print("Test  Epoch({}): Acc:{} Prec:{} Recall:{} F1-score:{} Loss:{}".format(epoch, Accuracy, Precision, Recall,
                                                                                 F1Score, Loss))

    return Accuracy, Precision, Recall, F1Score, Loss


def train(opt, device):
    print("start training\n")
    batch_size = opt.batch_size
    lr = opt.lr
    input_size = opt.input_size
    epoch = opt.epoch
    train_path = opt.train_path
    test_path = opt.val_path
    weight = opt.weight
    class_num = opt.class_num

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize(120),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}

    train_data = torchvision.datasets.ImageFolder(root=train_path, transform=data_transform["train"])
    test_data = torchvision.datasets.ImageFolder(root=test_path, transform=data_transform["test"])

    traindata = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                           num_workers=0)  # 将训练数据以每次n张图片的形式抽出进行训练
    testdata = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0,
                          pin_memory=True)  # 将训练数据以每次n张图片的形式抽出进行测试

    train_size = len(train_data)  # 训练集的长度
    test_size = len(test_data)  # 测试集的长度

    print("using {} images for training, {} images for validation.".format(train_size, test_size))  # 用于打印总的训练集数量和验证集数量

    model = vgg(num_classes=class_num).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    output_path = opt.output_path
    if os.path.exists(output_path) is not True:
        os.makedirs(output_path)
    log_eval_path = opt.log_eval
    if os.path.exists(log_eval_path):  # 如果log_eval.txt在存储之前存在则删除，防止后续内容冲突
        os.remove(log_eval_path)

    item_list = train_data.class_to_idx  # 获取类别名称以及对应的索引
    cla_dict = dict((val, key) for key, val in item_list.items())  # 将上面的键值对位置对调一下

    json_str = json.dumps(cla_dict, indent=4)  # 把类别和对应的索引写入根目录下class_indices.json文件中
    with open('output/class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_loss = []  # 存放训练集损失的数组
    train_accur = []  # 存放训练集准确率的数组
    test_loss = []  # 存放测试集损失的数组
    test_accur = []  # 存放测试集准确率的数组
    best_accur = 0.0  # 最高准确度

    for epoch in range(epoch):
        train_eval = train_epoch(epoch, model, traindata, criterion, optimizer, device)
        test_eval = test_epoch(epoch, model, testdata, criterion, device)

        # 画出eval的折线图利用plot
        train_loss.append(train_eval[4])
        test_loss.append(test_eval[4])
        train_accur.append(train_eval[0])
        test_accur.append(test_eval[0])

        # plot_eval(epoch, train_loss, test_loss, train_accur, test_accur, output_path)

        # 保存模型，保存测试集acc最高的模型和最后训练过程中最后一步的模型
        last_path = os.path.join(output_path, "last.pth")
        torch.save(model.state_dict(), last_path)

        if test_eval[0] >= best_accur:
            best_path = os.path.join(output_path, "best.pth")
            torch.save(model.state_dict(), best_path)
            best_accur = test_eval[0]

        # 保存log_eval.txt，包括best_model的acc（test）
        with open(log_eval_path, "a") as fp:
            fp.write("======\n")
            fp.write("Train Epoch({}): Acc:{} Prec:{} Recall:{} F1-score:{} Loss:{}\n".format(epoch, train_eval[0],
                                                                                              train_eval[1],
                                                                                              train_eval[2],
                                                                                              train_eval[3],
                                                                                              train_eval[4]))
            fp.write("Test  Epoch({}): Acc:{} Prec:{} Recall:{} F1-score:{} Loss:{}\n".format(epoch, test_eval[0],
                                                                                              test_eval[1],
                                                                                              test_eval[2],
                                                                                              test_eval[3],
                                                                                              test_eval[4]))
            fp.write("Best Acc(Test):{}\n".format(best_accur))
    # 下面的是画图过程，将上述存放的列表  画出来即可
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(epoch), train_loss,
             "ro-", label="Train loss")
    plt.plot(range(epoch), test_loss,
             "bs-", label="test loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(range(epoch), train_accur,
             "ro-", label="Train accur")
    plt.plot(range(epoch), test_accur,
             "bs-", label="test accur")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()
    plt.imsave('output/val.png')


def main():
    parse = argparse.ArgumentParser(description="classification")
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--lr", type=int, default=0.001)
    parse.add_argument("--input_size", type=int, default=120)
    parse.add_argument("--epoch", type=int, default=50)
    parse.add_argument("--weight", type=str, default='')
    parse.add_argument("--log_eval", type=str, default="output/vgg/log_val.txt")
    parse.add_argument("--train_path", type=str, default="data/train")
    parse.add_argument("--val_path", type=str, default="data/val")
    parse.add_argument("--class_num", type=int, default=5)
    parse.add_argument("--output_path", type=str, default="output/vgg/")

    opt = parse.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train(opt, device)

    return 0


if __name__ == '__main__':
    main()
