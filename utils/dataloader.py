import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import cv2
from PIL import Image
import os

classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def image_label(path):
    # 读取train.txt / test.txt
    # 得到每一个image的路径及其对应的标签，以列表形式返回
    image_list = []
    label_list = []
    
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            image = line.split(" ")[0]
            label = line.split(" ")[-1].split("\n")[0]
            label = classes.index(label)

            image_list.append(image)
            label_list.append(label)
            
    return image_list, label_list

class LoadImage(Dataset):
    def __init__(self, data_transform, trainval="/Users/hanxu/Documents/Code/git_code/computer-vision/图像分类/Image-Classification/trainval/myself/train.txt"):
        super(LoadImage, self).__init__()

        self.trainval = trainval
        self.data_transform = data_transform
        self.image_list, self.label_list = image_label(self.trainval)

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, item):
        image = self.image_list[item]
        label = self.label_list[item]
        # print("测试", image)
        # image = cv2.imread(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 用opencv读取需要转成RGB图像
        # image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_CUBIC)
        image = Image.open(image).convert('RGB')
        # print(image.size)
        if self.data_transform:
            image = self.data_transform(image)

        return image, label

if __name__ == "__main__":
    print("Test LoadImage......")
    # read_txt()
    transform = transforms.Compose([
                    transforms.Resize([224, 224]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    input_size = (224, 224)
    batch_size = 1
    train_datasets = LoadImage(transform)
    train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=False, num_workers=0)
    for x,y in train_dataloader:
        print(x.shape, y)

    
    
