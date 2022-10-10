import torch
from PIL import Image
from torch import nn
from torchvision.transforms import transforms
from modules.VGG import vgg
from modules.Resnet import res


def main():
    image_path = "data/pred_in/4.jpg"  # 相对路径 导入图片
    trans = transforms.Compose([
        transforms.Resize((120, 120)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])  # 将图片缩放为跟训练集图片的大小一样 方便预测，且将图片转换为张量
    image = Image.open(image_path)  # 打开图片
    image = image.convert("RGB")  # 将图片转换为RGB格式
    image = trans(image)  # 上述的缩放和转张量操作在这里实现
    # print(image)   #查看转换后的样子
    # image = torch.unsqueeze(image, dim=0)  #将图片维度扩展一维
    image = image.unsqueeze(0)

    classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']  # 预测种类

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 将代码放入GPU进行训练
    print("using {} device.".format(device))

    net = res(num_classes=5)  # 将模型命名为net,使用vgg
    net.load_state_dict(torch.load("output/Res/best.pth"))
    model = net.to(device)
    model.eval()
    # model.eval()  #关闭梯度，将模型调整为测试模式
    # exit()
    with torch.no_grad():  # 梯度清零
        outputs = model(image.to(device))  # 将图片打入神经网络进行测试
        # print(model)  #输出模型结构
        print(outputs)  # 输出预测的张量数组
        ans = (outputs.argmax(1)).item()  # 最大的值即为预测结果，找出最大值在数组中的序号，
        # 对应找其在种类中的序号即可然后输出即为其种类
        print(classes[ans])


if __name__ == '__main__':
    main()
