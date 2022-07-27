import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, drop_out_rate, num_input, bn_size, grow_rate):
        super(Bottleneck, self).__init__()
        self.drop_out_rate = drop_out_rate
        self.num_input = num_input
        self.feature = nn.Sequential(
            nn.BatchNorm2d(num_input),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_input, out_channels=bn_size * grow_rate, kernel_size=(1, 1), stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(bn_size * grow_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bn_size * grow_rate, out_channels=grow_rate, kernel_size=(3, 3), stride=1, padding=1,
                      bias=False),
        )
        self.drop_out = nn.Dropout(p=self.drop_out_rate)

    def forward(self, x):
        # print(self.num_input)
        y = self.feature(x)
        if self.drop_out_rate > 0:
            y = self.drop_out(y)

        return torch.cat([x, y], dim=1)


class Translation(nn.Module):
    def __init__(self, num_input, num_output):
        super(Translation, self).__init__()
        self.feature = nn.Sequential(
            nn.BatchNorm2d(num_input),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_input, out_channels=num_output, kernel_size=(1, 1), stride=1, padding=0,
                      bias=False),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        )

    def forward(self, x):
        x = self.feature(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input, bn_size, grow_rate, drop_out_rate=0):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_layers):
            layer.append(
                Bottleneck(drop_out_rate=drop_out_rate, num_input=num_input + i * grow_rate, bn_size=bn_size,
                           grow_rate=grow_rate))
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)


class DenseNet(nn.Module):
    def __init__(self, init_num=112, grow_rate=32, block=(6, 12, 24, 16), bn_size=4, drop_out_rate=0, num_classes=5):
        super(DenseNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=init_num, kernel_size=(7, 7), stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_num),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0)
        )
        feature_num = init_num
        self.layer1 = DenseBlock(num_layers=block[0], num_input=feature_num, bn_size=bn_size, grow_rate=grow_rate,
                                 drop_out_rate=drop_out_rate)
        feature_num = feature_num + grow_rate * block[0]
        # print(feature_num)
        self.translation1 = Translation(num_input=feature_num, num_output=feature_num // 2)
        feature_num = feature_num // 2
        # print(feature_num)

        self.layer2 = DenseBlock(num_layers=block[1], num_input=feature_num, bn_size=bn_size, grow_rate=grow_rate,
                                 drop_out_rate=drop_out_rate)
        feature_num = feature_num + grow_rate * block[1]
        # print(feature_num)
        self.translation2 = Translation(num_input=feature_num, num_output=feature_num // 2)
        feature_num = feature_num // 2
        # print(feature_num)

        self.layer3 = DenseBlock(num_layers=block[2], num_input=feature_num, bn_size=bn_size, grow_rate=grow_rate,
                                 drop_out_rate=drop_out_rate)
        # print(feature_num)
        feature_num = feature_num + grow_rate * block[2]
        # print(feature_num)
        self.translation3 = Translation(num_input=feature_num, num_output=feature_num // 2)
        feature_num = feature_num // 2
        # print(feature_num)

        self.layer4 = DenseBlock(num_layers=block[3], num_input=feature_num, bn_size=bn_size, grow_rate=grow_rate,
                                 drop_out_rate=drop_out_rate)
        feature_num = feature_num + grow_rate * block[3]
        # print(feature_num)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # feature_num = feature_num // 7
        # print(feature_num)
        self.fc = nn.Linear(feature_num, num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = self.layer1(x)
        x = self.translation1(x)
        x = self.layer2(x)
        x = self.translation2(x)
        x = self.layer3(x)
        x = self.translation3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


cfgs = {
    'DenseNet121': (6, 12, 24, 16),
    'DenseNet169': (6, 12, 32, 32),
}


def dense(model_name="DenseNet169", num_classes=5, **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]
    model = DenseNet(block=cfg, num_classes=num_classes, **kwargs)

    return model
