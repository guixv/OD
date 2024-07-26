# -*- coding: utf-8 -*-
"""
Created on 2021/8/9 16:52
@author: Janben
参考链接：
1. 交叉熵：https://blog.csdn.net/qq_41805511/article/details/99438838
2. KLDivLoss: https://blog.csdn.net/qq_36533552/article/details/104034759
3. LabelSmoothing: https://blog.csdn.net/qq_36560894/article/details/118424356
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Function
from torch.autograd import Variable
import warnings
warnings.simplefilter('ignore')
 
class MyCrossEntropyLoss():
 
    def __init__(self, weight=None, size_average=True):
        """
        初始化参数，因为要实现 torch.nn.CrossEntropyLoss 的两个比较重要的参数
        :param weight: 给予每个类别不同的权重
        :param size_average: 是否要对 loss 求平均
        """
 
        self.weight = weight
        self.size_average = size_average
 
 
    def __call__(self, input, target):
        """
        计算损失
        这个方法让类的实例表现的像函数一样，像函数一样可以调用
        :param input: (batch_size, C)，C是类别的总数
        :param target: (batch_size, 1)
        :return: 损失
        """
 
        batch_loss = 0.
        for i in range(input.shape[0]):
            # print('***',input[i, target[i]],i,target[i],np.exp(input[i, :]))
            numerator = torch.exp(input[i, target[i]])     # 分子
            denominator = torch.sum(torch.exp(input[i, :]))   # 分母
 
            # 计算单个损失
            loss = -torch.log(numerator / denominator)
            if self.weight:
                loss = self.weight[target[i]] * loss
            print("单个损失： ",loss)
 
            # 损失累加
            batch_loss += loss
 
        # 整个 batch 的总损失是否要求平均
        if self.size_average == True:
            batch_loss /= input.shape[0]
 
        return batch_loss
 
 
class MyKLDivLossFunc(nn.Module):
    def __init__(self,reduce = True):
        super(MyKLDivLossFunc,self).__init__()
        self.reduce = reduce
 
    def forward(self,x,target):
        logtarget = torch.log(target+0.00001)  #加一个非常小的数，防止当target中有0时log得到-inf
        loss = target*(logtarget-x)
        if self.reduce == False:
            return loss
        else:
            return torch.sum(loss)
 
class LabelSmoothingLoss(nn.Module):
    "Implement label smoothing."
 
    def __init__(self, class_num, smoothing):
        '''
        :param class_num: 有5个类别，那么class_num=5
        :param smoothing: 标签平滑的程度，为0时表示不进行标签平滑
        '''
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.class_num = class_num
        # self.criterion = nn.KLDivLoss(size_average=True)
        # self.criterion = nn.KLDivLoss(size_average=False)
 
    def forward(self, x, target):
        '''
        :param x: 预测结果，形状为(batchsize,classnum)
        :param target: 真实标签，形状为(batchsize,)
        :return:
        '''
        # print(x.shape)
        assert x.size(1) == self.class_num
        # if self.smoothing <=0.0 or self.smoothing == None:
        if self.smoothing == None:
            return nn.CrossEntropyLoss()(x,target)
 
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.class_num-1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)  #此行代码实现了标签平滑
        #计算交叉熵，与nn.CrossEntropyLoss()公式一样，所以当smoothing=0.0时，输出的损失值与nn.CrossEntropyLoss()的一样的
        logprobs = Function.log_softmax(x,dim=1)  #softmax+log
        mean_loss = -torch.sum(true_dist*logprobs)/x.size(-2)  #平均损失，所以要除以样本数量
        # return mean_loss,true_dist
        return mean_loss
 
if __name__ == "__main__":
    input = np.array([[-1.5616, -0.7906,  1.4143, -0.0957,  0.1657],
        [-1.4285,  0.3045,  1.5844, -2.1508,  1.8181],
        [ 1.0205, -1.3493, -1.2965,  0.1715, -1.2118]])
    target = np.array([2, 0, 3])
    test_input = torch.from_numpy(input)
    test_target = torch.from_numpy(target).long()
    #自定义的交叉熵函数
    criterion = MyCrossEntropyLoss()
    # 类中实现了 __call__，所以类实例可以像函数一样可以调用
    loss = criterion(test_input, test_target)
    print("+++My_CrossEntroy： ", loss) #输出：  tensor(1.9606, dtype=torch.float64)
 
    #torch.nn中库函数
    #The `input` is expected to contain raw, unnormalized scores for each class.
    #Input: 形状为(N, C)` where `C = number of classes`，N是batchsize
    #交叉熵的input不需要进行任何标准化（不需要softmax，不需要log)，用原始的数据
    #Target: :math:`(N)`
    test_loss = nn.CrossEntropyLoss()
    test_out = test_loss(test_input,test_target)
    print('+++Normal CrossEntroy:',test_out) #test loss: tensor(1.9606, dtype=torch.float64)
 
    print('+'*50)
    lloss = LabelSmoothingLoss(5,smoothing=0.1)
    loss_result, true_dist = lloss.forward(test_input,test_target)
    print('label smoothing loss result:',loss_result)  #label smoothing loss result: tensor(2.2265, dtype=torch.float64)
    lloss = LabelSmoothingLoss(5,smoothing=0.)
    loss_result,_ = lloss.forward(test_input,test_target)
    print('label smoothing loss result:',loss_result)
 
 
    print('-' * 50)  #以下是验证自定义的KLDivLoss的正确性
    #以下的test_input要理解为已经完成log运算之后的数据
    print('normal kld loss:\n\t\t', nn.KLDivLoss(size_average=True, reduce=True)(test_input, Variable(true_dist, requires_grad=False)))
    print('normal kld loss(default):\n\t\t',nn.KLDivLoss()(test_input,Variable(true_dist,requires_grad=False))) #默认size_average=True, reduce=True
    print('normal kld loss:\n\t\t', nn.KLDivLoss(size_average=False, reduce=True)(test_input, Variable(true_dist, requires_grad=False)))
    print('my kld loss:\n\t\t', MyKLDivLossFunc(reduce=True)(test_input, true_dist)) #自己实现的不进行size_average,size_average是指除以元素个数，本例中元素个数为15
    print('-'*50)
    print('normal kld loss:\n\t\t', nn.KLDivLoss(size_average=True, reduce=False)(test_input, Variable(true_dist, requires_grad=False)))
    print('normal kld loss:\n\t\t', nn.KLDivLoss(size_average=False, reduce=False)(test_input, Variable(true_dist, requires_grad=False)))
    print('my kld loss:\n\t\t', MyKLDivLossFunc(reduce=False)(test_input, true_dist))
 
    p = torch.Tensor([[0,0.1,0.3],[0.1,0.9,0.3],[0,0.1,0.]])
    t = torch.Tensor([[0,0,1.],[0,1.,0],[1.,0,0]])
    c = nn.KLDivLoss(size_average=False)
    print(c(p,t))
    print(MyKLDivLossFunc()(p,t))
    print(torch.log(t+0.00001))
