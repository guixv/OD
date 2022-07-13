import torch
import torchvision
import torchvision.models
import json
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

print(torch.__version__)

data_transform = {
    "train": transforms.Compose([transforms.Resize((120, 120)),
        # transforms.RandomResizedCrop(120),
        #                          transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize((120, 120)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

train_data = torchvision.datasets.ImageFolder(root="./data/train", transform=data_transform["train"])
test_data = torchvision.datasets.ImageFolder(root="./data/val", transform=data_transform["val"])

traindata = DataLoader(dataset=train_data, batch_size=4, shuffle=True, num_workers=0)  # 将训练数据以每次n张图片的形式抽出进行训练
testdata = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0)  # 将训练数据以每次n张图片的形式抽出进行测试


train_size = len(train_data)  # 训练集的长度
test_size = len(test_data)  # 测试集的长度
# print(train_size)  # 输出训练集长度看一下，相当于看看有几张图片
# print(test_size)  # 输出测试集长度看一下，相当于看看有几张图片
print("using {} images for training, {} images for validation.".format(train_size, test_size))  # 用于打印总的训练集数量和验证集数量


item_list = train_data.class_to_idx  # 获取类别名称以及对应的索引
cla_dict = dict((val, key) for key, val in item_list.items())  # 将上面的键值对位置对调一下

json_str = json.dumps(cla_dict, indent=4)  # 把类别和对应的索引写入根目录下class_indices.json文件中
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
print("using {} device.".format(device))


class VGG(nn.Module):
    def __init__(self, features, num_classes=5, init_weights=True):
        super(VGG, self).__init__()
        # self.features = features
        # 'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        self.stage01 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.stage02 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.stage03 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.stage04 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.stage05 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(4608, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()  # 参数初始化

    def forward(self, x):
        # N x 3 x 224 x 224
        # x = self.features(x)
        # N x 512 x 7 x 7
        x = self.stage01(x)
        x = self.stage02(x)
        x = self.stage03(x)
        x = self.stage04(x)
        x = self.stage05(x)
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():  # 遍历各个层进行参数初始化
            if isinstance(m, nn.Conv2d):  # 如果是卷积层的话 进行下方初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_uniform_(m.weight)  # 正态分布初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 如果偏置不是0 将偏置置成0  相当于对偏置进行初始化
            elif isinstance(m, nn.Linear):  # 如果是全连接层
                # nn.init.xavier_uniform_(m.weight)  # 也进行正态分布初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  # 将所有偏执置为0


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(3, 3), padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg11", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model


VGGnet = vgg(num_classes=5, init_weights=True)  # 将模型命名为VGGnet
VGGnet.to(device)
print(VGGnet.to(device))  # 输出模型结构

test1 = torch.ones(64, 3, 120, 120)  # 测试一下输出的形状大小 输入一个64,3,120,120的向量

test1 = VGGnet(test1.to(device))  # 将向量打入神经网络进行测试
print(test1.shape)  # 查看输出的结果

epoch = 5  # 迭代次数即训练次数
learning = 0.01  # 学习率
optimizer = torch.optim.Adam(VGGnet.parameters(), lr=learning)  # 使用Adam优化器-写论文的话可以具体查一下这个优化器的原理
loss = nn.CrossEntropyLoss()  # 损失计算方式，交叉熵损失函数

train_loss_all = []  # 存放训练集损失的数组
train_accur_all = []  # 存放训练集准确率的数组
test_loss_all = []  # 存放测试集损失的数组
test_accur_all = []  # 存放测试集准确率的数组
for i in range(epoch):  # 开始迭代
    train_loss = 0  # 训练集的损失初始设为0
    train_num = 0.0  #
    train_accuracy = 0.0  # 训练集的准确率初始设为0
    VGGnet.train()  # 将模型设置成 训练模式
    train_bar = tqdm(traindata)  # 用于进度条显示，没啥实际用处
    for step, data in enumerate(train_bar):  # 开始迭代跑， enumerate这个函数不懂可以查查，将训练集分为 data是序号，data是数据
        img, target = data  # 将data 分位 img图片，target标签
        optimizer.zero_grad()  # 清空历史梯度
        outputs = VGGnet(img.to(device))  # 将图片打入网络进行训练,outputs是输出的结果
        # print("[Test outoputs]:", outputs)

        loss1 = loss(outputs, target.to(device))  # 计算神经网络输出的结果outputs与图片真实标签target的差别-这就是我们通常情况下称为的损失
        outputs = torch.argmax(outputs, 1)  # 会输出10个值，最大的值就是我们预测的结果 求最大值
        loss1.backward()  # 神经网络反向传播
        optimizer.step()  # 梯度优化 用上面的abam优化
        train_loss += abs(loss1.item()) * img.size(0)  # 将所有损失的绝对值加起来
        accuracy = torch.sum(outputs == target.to(device))  # outputs == target的 即使预测正确的，统计预测正确的个数,从而计算准确率
        train_accuracy = train_accuracy + accuracy  # 求训练集的准确率
        train_num += img.size(0)  #

    print("epoch：{} ， train-Loss：{} , train-accuracy：{}".format(i + 1, train_loss / train_num,  # 输出训练情况
                                                                train_accuracy / train_num))

    train_loss_all.append(train_loss / train_num)  # 将训练的损失放到一个列表里 方便后续画图
    train_accur_all.append(train_accuracy.double().item() / train_num)  # 训练集的准确率
    test_loss = 0  # 同上 测试损失
    test_accuracy = 0.0  # 测试准确率
    test_num = 0
    VGGnet.eval()  # 将模型调整为测试模型
    with torch.no_grad():  # 清空历史梯度，进行测试  与训练最大的区别是测试过程中取消了反向传播
        test_bar = tqdm(testdata)
        for data in test_bar:
            img, target = data

            outputs = VGGnet(img.to(device))
            loss2 = loss(outputs, target.to(device))
            outputs = torch.argmax(outputs, 1)
            test_loss = test_loss + abs(loss2.item()) * img.size(0)
            accuracy = torch.sum(outputs == target.to(device))
            test_accuracy = test_accuracy + accuracy
            test_num += img.size(0)

    print("test-Loss：{} , test-accuracy：{}".format(test_loss / test_num, test_accuracy / test_num))
    test_loss_all.append(test_loss / test_num)
    test_accur_all.append(test_accuracy.double().item() / test_num)
    torch.save(VGGnet.state_dict(), "VGG_new.pth")

# 下面的是画图过程，将上述存放的列表  画出来即可
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(epoch), train_loss_all,
         "ro-", label="Train loss")
plt.plot(range(epoch), test_loss_all,
         "bs-", label="test loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.subplot(1, 2, 2)
plt.plot(range(epoch), train_accur_all,
         "ro-", label="Train accur")
plt.plot(range(epoch), test_accur_all,
         "bs-", label="test accur")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()
plt.show()

torch.save(VGGnet.state_dict(), "VGG_new.pth")
print("模型已保存")
