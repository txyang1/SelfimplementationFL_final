#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from models.LDP import LDPModule


#这段代码定义了一个名为 DatasetSplit 的自定义数据集类，用于创建一个子数据集（用户的样本集），该子数据集包含原始数据集中特定索引集合的样本
class DatasetSplit(Dataset):#定义了一个名为 DatasetSplit 的类，该类继承自 PyTorch 的 Dataset 类，用于创建自定义数据集
    def __init__(self, dataset, idxs):#初始化方法，接受两个参数：dataset 是原始数据集对象，
        #idxs 是子数据集的索引集合，idxs=dict_users[idx]是第idx个用户被分配的图像索引的集合，{n0, n1, n2, n3, n4, ...}第n个用户。
        self.dataset = dataset
        self.idxs = list(idxs)#将索引集合转换为列表，并存储在实例变量 self.idxs 中。

    def __len__(self):
        return len(self.idxs)#返回子数据集索引集合的长度，即子数据集的样本数量，即单个用户的样本数量，假设为60个

    def __getitem__(self, item):#用于获取数据集中特定索引的样本，即获取单个用户的样本中某一个索引
        image, label = self.dataset[self.idxs[item]]#根据子数据集的索引集合中的第 item 个索引，从原始数据集中获取相应的图像和标签。
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        #初始化方法，接受三个参数：args 是用于配置训练的参数对象，dataset 是原始数据集对象，idxs 是本地客户端的索引集合。
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()#定义交叉熵损失函数，用于计算损失
        self.selected_clients = []#初始化一个空列表 selected_clients，用于存储选择的客户端。
        #创建一个数据加载器 ldr_train，该加载器使用了 DatasetSplit 类来生成一个子数据集，该子数据集仅包含本地客户端的样本。
        #这个数据加载器将用于在本地训练时迭代访问数据，local_bs=10, 即从客户端样本集600个中，每10个迭代访问，一共60批
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()#将模型设置为训练模式
        # train and update
        #定义随机梯度下降（SGD）优化器，用于更新模型参数。
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []#初始化一个列表，用于存储每个训练周期的损失
        #进行梯度扰动
        #ldp = LDPModule(alpha=1,c=1,rho=2)

        for iter in range(self.args.local_ep):#对每个训练周期进行迭代，5次#default 5 E
            batch_loss = []#初始化一个列表，用于存储每个批次的损失。
            for batch_idx, (images, labels) in enumerate(self.ldr_train):#对本地数据加载器中的每个批次进行迭代。
                '''这段代码是一个用于遍历训练数据集的循环。
                它使用了 Python 中的 enumerate 函数来同时获取批次的索引和对应的数据。
                在每次迭代中，会从 self.ldr_train 数据加载器中获取一个批次的数据，并将图像数据和对应的标签解包到 images 和 labels 中。
                下面是对代码的详细解释：
                for batch_idx, (images, labels) in enumerate(self.ldr_train):：这是一个 for 循环，
                enumerate(self.ldr_train) 会在每次迭代中返回数据加载器 self.ldr_train 中的一个批次数据，
                并同时获取批次的索引 batch_idx 和批次数据 (images, labels)。
                batch_idx：批次的索引，表示当前批次在整个数据集中的位置。所以按照假设有60批
                (images, labels)：批次数据的元组，包含了当前批次的图像数据和对应的标签。'''
                images, labels = images.to(self.args.device), labels.to(self.args.device)#将图像和标签移动到设备上（通常是 GPU）
                net.zero_grad()#清零模型的梯度
                log_probs = net(images)#将图像输入模型，并获取预测的对数概率
                loss = self.loss_func(log_probs, labels)#计算预测结果与真实标签之间的交叉熵损失。
                loss.backward()#反向传播，计算梯度
                optimizer.step()#更新模型参数
                if self.args.verbose and batch_idx % 10 == 0:#如果设置了 verbose 并且批次索引是 10 的倍数。
                                                             #verbose 是一个参数，用于控制训练过程中是否输出详细的信息。
                    #通常，当 verbose 设置为 True 时，会在训练过程中输出额外的信息，例如每个训练周期的损失值以及每个批次的训练状态。
                    #print(batch_idx,len(self.ldr_train),len(images)) #50 60 10
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))#打印当前训练状态，包括训练周期iter=0-4、批次索引batch_idx=0-60、总批次数60和损失值loss.item。
                batch_loss.append(loss.item())#将当前批次的损失值添加到 batch_loss 列表中
            epoch_loss.append(sum(batch_loss)/len(batch_loss))#计算并存储当前训练周期的平均损失。
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)#返回训练结束后的模型参数字典和训练周期的平均损失。
                                                   #train 方法内部确保了 w 与 net.state_dict() 保持同步，那么它们可能是相等的
    '''net,state_dict() 方法返回的字典包含了模型中所有可学习参数的当前值。这些参数通常是模型中的权重和偏置项，用于在训练过程中更新模型的参数。
             {
              'layer1.weight': tensor([...]),  # 第一个层的权重
              'layer2.weight': tensor([...]),  # 第二个层的权重
                ...
            }'''