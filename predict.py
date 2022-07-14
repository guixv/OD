import torch
from PIL import Image
from torch import nn
from torchvision.transforms import transforms


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

#以上是神经网络结构，因为读取了模型之后代码还得知道神经网络的结构才能进行预测



if __name__ == '__main__':
    image_path = "data/pred_in/5.jpg"#相对路径 导入图片
    trans = transforms.Compose([
       transforms.Resize((120 , 120)),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])   #将图片缩放为跟训练集图片的大小一样 方便预测，且将图片转换为张量
    image = Image.open(image_path)  #打开图片
    image = image.convert("RGB")  #将图片转换为RGB格式
    image = trans(image)   #上述的缩放和转张量操作在这里实现
    # print(image)   #查看转换后的样子
    # image = torch.unsqueeze(image, dim=0)  #将图片维度扩展一维
    image = image.unsqueeze(0)


    classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']  #预测种类

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #将代码放入GPU进行训练
    print("using {} device.".format(device))

    VGGnet = vgg(num_classes=5, init_weights=False)  # 将模型命名为VGGnet
    VGGnet.load_state_dict(torch.load("output/best.pth"))
    model = VGGnet.to(device)
    model.eval()
    # model.eval()  #关闭梯度，将模型调整为测试模式
    # exit()
    with torch.no_grad():  #梯度清零
        outputs = model(image.to(device))  #将图片打入神经网络进行测试
        # print(model)  #输出模型结构
        print(outputs)  #输出预测的张量数组
        ans = (outputs.argmax(1)).item()  #最大的值即为预测结果，找出最大值在数组中的序号，
        # 对应找其在种类中的序号即可然后输出即为其种类
        print(classes[ans])