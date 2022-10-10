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

from utils.utils import Evaluation
import argparse
import numpy as np
import os
from modules.MNISTmodule import vgg

def train_epoch(epoch, model, traindata, criterion, optimizer, device,scheduler):
    model.train()
    true = []
    pre = []
    losses = 0
    for image, label in tqdm(traindata):
        image, label = Variable(image.float()).to(device), Variable(label).to(device)

        optimizer.zero_grad()  # 初始化梯度值
        output = model(image)
        # print(output)
        # print(label)

        loss = criterion(output, label)
        loss.backward()  # 反向求解梯度
        losses += loss.item()
        optimizer.step()  # 更新参数
        # scheduler.step()  #更新学习率

        pre_ = torch.argmax(output, 1)
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


def plot_draw(train_loss,test_loss,train_accur,test_accur,epoch,output_path):
    # 下面的是画图过程，将上述存放的列表  画出来即可
    print(range(epoch + 1))
    print(train_loss)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(epoch + 1), train_loss,
             "ro-", label="Train loss")
    plt.plot(range(epoch + 1), test_loss,
             "bs-", label="test loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(range(epoch + 1), train_accur,
             "ro-", label="Train accur")
    plt.plot(range(epoch + 1), test_accur,
             "bs-", label="test accur")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig(output_path + 'val.png')
    plt.show()

def train(opt, device):
    print("start training\n")
    batch_size = opt.batch_size
    lr = opt.lr
    epoch = opt.epoch
    train_path = opt.train_path
    test_path = opt.val_path
    weight = opt.weight
    lrf = opt.lrf
    class_num = opt.class_num

    model = vgg(num_classes=class_num).to(device)
    if weight != "":
        assert os.path.exists(weight), "weights file: '{}' not exist.".format(weight)
        weights_dict = torch.load(weight, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))



    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    traindata = train_loader
    testdata=test_loader
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    # print(example_targets)
    # print(example_data.shape)

    # fig = plt.figure()
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1)
    #     plt.tight_layout()
    #     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    #     plt.title("Ground Truth: {}".format(example_targets[i]))
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()



    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # 优化器

    lf = lambda x: ((1 + math.cos(x * math.pi / (epoch + 1))) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 学习率变化

    output_path = opt.output_path
    if os.path.exists(output_path) is not True:
        os.makedirs(output_path)
    log_eval_path = opt.output_path + "/log_eval.txt"
    if os.path.exists(log_eval_path):  # 如果log_eval.txt在存储之前存在则删除，防止后续内容冲突
        os.remove(log_eval_path)

    log_list_path = opt.output_path + "/log_list.txt"
    if os.path.exists(log_list_path):  # 如果log_eval.txt在存储之前存在则删除，防止后续内容冲突
        os.remove(log_list_path)


    train_loss = []  # 存放训练集损失的数组
    train_accur = []  # 存放训练集准确率的数组
    test_loss = []  # 存放测试集损失的数组
    test_accur = []  # 存放测试集准确率的数组
    best_accur = 0.0  # 最高准确度

    with open(log_list_path, "a") as fp:
        fp.write("epoch,train_accur,test_accur,,train_loss,test_loss\n")
    for epoch in range(epoch):
        train_eval = train_epoch(epoch, model, traindata, criterion, optimizer, device, scheduler)
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

        # plot_draw(train_loss, test_loss, train_accur, test_accur, epoch, output_path)
        with open(log_list_path, "a") as fp:
            fp.write("{},{},{},,{},{}\n".format(epoch, train_eval[0], test_eval[0], train_eval[4], test_eval[4]))
    # 下面的是画图过程，将上述存放的列表  画出来即可
    plot_draw(train_loss, test_loss, train_accur, test_accur, epoch, output_path)




def main():
    parse = argparse.ArgumentParser(description="classification")
    parse.add_argument("--batch_size", type=int, default=4)
    parse.add_argument("--lr", type=int, default=0.001)
    parse.add_argument("--lrf", type=int, default=0.01)
    parse.add_argument("--epoch", type=int, default=5)
    parse.add_argument("--weight", type=str, default="")
    # parse.add_argument("--log_eval", type=str, default="output/ViT/log_val.txt")
    # parse.add_argument("--log_list", type=str, default="output/ViT/log_list.txt")
    parse.add_argument("--train_path", type=str, default="data/train")
    parse.add_argument("--val_path", type=str, default="data/val")
    parse.add_argument("--class_num", type=int, default=10)
    parse.add_argument("--output_path", type=str, default="output/MNIST/")

    opt = parse.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train(opt, device)

    return 0


if __name__ == '__main__':
    main()


