import torch
import torch.nn as nn
from models.odconv import ODConv2d

def odconv3x3(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num)


def odconv1x1(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                    reduction=reduction, kernel_num=kernel_num)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, width=64, down_sample=None):
        super(Bottleneck, self).__init__()
        width = 64
        self.conv1 = odconv1x1(in_channel, width, stride=1)
        self.conv2 = odconv3x3(width, width,stride=stride)
        self.conv3 = odconv1x1(width, out_channel * self.expansion, stride=1)
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
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

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

    def net_update_temperature(self, temperature):
        for m in self.modules():
            if hasattr(m, "update_temperature"):
                m.update_temperature(temperature)

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


def res(model_name="resnet50", num_classes=5, **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]
    model = ResNet(Bottleneck, cfg, num_classes, **kwargs)

    return model
