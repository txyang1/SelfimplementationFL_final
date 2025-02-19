import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random


class Client(object):
    def __init__(self, args, trainloader):
        #初始化方法，接受三个参数：args 是用于配置训练的参数对象，dataset 是原始数据集对象，idxs 是本地客户端的索引集合。
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()#定义交叉熵损失函数，用于计算损失
        self.selected_clients = []#初始化一个空列表 selected_clients，用于存储选择的客户端。
        #创建一个数据加载器 ldr_train，该加载器使用了 DatasetSplit 类来生成一个子数据集，该子数据集仅包含本地客户端的样本。
        #这个数据加载器将用于在本地训练时迭代访问数据，local_bs=10, 即从客户端样本集600个中，每10个迭代访问，一共60批
        self.trainloader = trainloader
        #self.trainloaders, self.valloaders, self.testloader = prepare_dataset(dataset_train,dataset_test,num_partitions=100, batch_size=128)

    def train(self, net ):
        net.train()#将模型设置为训练模式
        # train and update
        #定义随机梯度下降（SGD）优化器，用于更新模型参数。
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []#初始化一个列表，用于存储每个训练周期的损失
        for iter in range(self.args.local_ep):#对每个训练周期进行迭代，5次#default 5 E
            batch_loss = []#初始化一个列表，用于存储每个批次的损失。
            for images, labels in self.trainloader:#对本地数据加载器中的每个批次进行迭代。
                images, labels = images.to(self.args.device), labels.to(self.args.device)#将图像和标签移动到设备上（通常是 GPU）
                optimizer.zero_grad()#清零模型的梯度
                log_probs = net(images)#将图像输入模型，并获取预测的对数概率
                loss = self.loss_func(log_probs, labels)#计算预测结果与真实标签之间的交叉熵损失。
                loss.backward()#反向传播，计算梯度
                optimizer.step()#更新模型参数
                batch_loss.append(loss.item())#将当前批次的损失值添加到 batch_loss 列表中
            epoch_loss.append(sum(batch_loss)/len(batch_loss))#计算并存储当前训练周期的平均损失。
        
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)#返回训练结束后的模型参数字典和训练周期的平均损失。
    
    '''def train(net, trainloader, optimizer, epochs):
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        net.train()
        epoch_loss = []
        for _ in range(epochs):
            batch_loss = []
            for images, labels in trainloader:
                optimizer.zero_grad()
                loss = criterion(net(images), labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net,sum(epoch_loss) / len(epoch_loss)'''