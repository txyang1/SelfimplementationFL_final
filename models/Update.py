#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset



class DatasetSplit(Dataset):#定义了一个名为 DatasetSplit 的类，该类继承自 PyTorch 的 Dataset 类，用于创建自定义数据集
    def __init__(self, dataset, idxs):#初始化方法，接受两个参数：dataset 是原始数据集对象，
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
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.nk = len(idxs)
    def train(self, net):
        net.train()#将模型设置为训练模式
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []#初始化一个列表，用于存储每个训练周期的损失
        for iter in range(self.args.local_ep):#对每个训练周期进行迭代，5次#default 5 E
            batch_loss = []#初始化一个列表，用于存储每个批次的损失。
            for batch_idx, (images, labels) in enumerate(self.ldr_train):#对本地数据加载器中的每个批次进行迭代。
                images, labels = images.to(self.args.device), labels.to(self.args.device)#将图像和标签移动到设备上（通常是 GPU）
                net.zero_grad()#清零模型的梯度
                log_probs = net(images)#将图像输入模型，并获取预测的对数概率
                loss = self.loss_func(log_probs, labels)#计算预测结果与真实标签之间的交叉熵损失。
                loss.backward()#反向传播，计算梯度
                optimizer.step()#更新模型参数
                batch_loss.append(loss.item())#将当前批次的损失值添加到 batch_loss 列表中
            epoch_loss.append(sum(batch_loss)/len(batch_loss))#计算并存储当前训练周期的平均损失。

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss),self.nk# 返回训练结束后的模型参数字典和训练周期的平均损失。


  
