import torch
from PIL import Image
from torch import nn
from torchvision.transforms import transforms

image_path = "1.png"#相对路径 导入图片
trans = transforms.Compose([transforms.Resize((120 , 120)),
                           transforms.ToTensor()])   #将图片缩放为跟训练集图片的大小一样 方便预测，且将图片转换为张量
image = Image.open(image_path)  #打开图片
print(image)  #输出图片 看看图片格式
image = image.convert("RGB")  #将图片转换为RGB格式
image = trans(image)   #上述的缩放和转张量操作在这里实现
print(image)   #查看转换后的样子
image = torch.unsqueeze(image, dim=0)  #将图片维度扩展一维

classes = ["1" , "2" , "3" , "4" , "5" , "6" , "7" ]  #预测种类
class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
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
            self._initialize_weights()   #参数初始化

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():         #遍历各个层进行参数初始化
            if isinstance(m, nn.Conv2d):   #如果是卷积层的话 进行下方初始化
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)  #正态分布初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)     #如果偏置不是0 将偏置置成0  相当于对偏置进行初始化
            elif isinstance(m, nn.Linear):        #如果是全连接层
                nn.init.xavier_uniform_(m.weight)    #也进行正态分布初始化
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  #将所有偏执置为0


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model
#以上是神经网络结构，因为读取了模型之后代码还得知道神经网络的结构才能进行预测
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #将代码放入GPU进行训练
print("using {} device.".format(device))


model = torch.load("VGG.pth")  #读取模型
model.eval()  #关闭梯度，将模型调整为测试模式
with torch.no_grad():  #梯度清零
    outputs = model(image.to(device))  #将图片打入神经网络进行测试
    print(model)  #输出模型结构
    print(outputs)  #输出预测的张量数组
    ans = (outputs.argmax(1)).item()  #最大的值即为预测结果，找出最大值在数组中的序号，
    # 对应找其在种类中的序号即可然后输出即为其种类
    print(classes[ans])