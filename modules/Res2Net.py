import torch
import torch.nn as nn


class Res2block(nn.Module):
    def __init__(self,in_channel,out_channel,scales=4):
        super(Res2block, self).__init__()

        if out_channel % scales != 0:  # 输出通道数为4的倍数
            raise ValueError('Planes must be divisible by scales')

        self.scales = scales
        # 1*1的卷积层
        self.inconv = nn.Sequential(
            nn.Conv2d(in_channel, 32, 1, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # 3*3的卷积层，一共有3个卷积层和3个BN层
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        # 1*1的卷积层
        self.outconv = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        input = x
        x = self.inconv(x)

        # scales个部分
        xs = torch.chunk(x, self.scales, 1)
        ys = []
        ys.append(xs[0])
        ys.append(self.conv1(xs[1]))
        ys.append(self.conv2(xs[2]) + ys[1])
        ys.append(self.conv2(xs[3]) + ys[2])
        y = torch.cat(ys, 1)

        y = self.outconv(y)

        output = y + input

        return output


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, width=64, down_sample=None):
        super(Bottleneck, self).__init__()
        width = 64
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=(1, 1), stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=(3, 3), stride=stride, bias=False,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=(1, 1), stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.bn2 = nn.BatchNorm2d(width)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.reLu = nn.ReLU(inplace=True)
        self.down_sample = down_sample

    def forward(self, x):
        identity = x
        if self.down_sample is not None:
            identity = self.down_sample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.reLu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.reLu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += identity

        x = self.reLu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block_type, block_num, num_classes=5, include_top=True):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.in_channel = 64
        self.block_type = block_type
        self.include_top = include_top
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=(7, 7), padding=3, stride=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.reLu = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.layer1 = self.make_layers(channel=64, block_type=block_type, block_num=block_num[0], stride=1)
        self.layer2 = self.make_layers(channel=128, block_type=block_type, block_num=block_num[1], stride=2)
        self.layer3 = self.make_layers(channel=256, block_type=block_type, block_num=block_num[2], stride=2)
        self.layer4 = self.make_layers(channel=512, block_type=block_type, block_num=block_num[3], stride=2)

        if self.include_top:  ##???
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block_type.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.reLu(x)
        x = self.maxPool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

    def make_layers(self, channel, block_type, block_num, stride=1):
        down_sample = None
        if stride != 1 or self.in_channel != channel * block_type.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block_type.expansion, kernel_size=(1, 1), stride=stride,
                          bias=False),
                nn.BatchNorm2d(channel * block_type.expansion),
            )

        layer = []
        layer.append(block_type(self.in_channel, channel, down_sample=down_sample, stride=stride, width=64))
        self.in_channel = block_type.expansion * channel

        for i in range(1, block_num):
            layer.append(block_type(self.in_channel, channel, width=64))

        return nn.Sequential(*layer)


cfgs = {
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
}


def res2(model_name="resnet50", num_classes=5, **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]
    model = ResNet(Res2block, cfg, num_classes, **kwargs)

    return model
