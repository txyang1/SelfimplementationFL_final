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

import torch


def train2(args, net, trainloader):
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()
    epoch_loss = []
    for iter in range(args.local_ep):  # 对每个训练周期进行迭代，5次#default 5 E
        batch_loss = []  # 初始化一个列表，用于存储每个批次的损失。
        for images, labels in trainloader:  # 对本地数据加载器中的每个批次进行迭代。
            images, labels = images.to(args.device), labels.to(args.device)  # 将图像和标签移动到设备上（通常是 GPU）
            net.zero_grad()  # 清零模型的梯度
            log_probs = net(images)  # 将图像输入模型，并获取预测的对数概率
            loss = criterion(log_probs, labels)  # 计算预测结果与真实标签之间的交叉熵损失。
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数
            batch_loss.append(loss.item())  # 将当前批次的损失值添加到 batch_loss 列表中
        epoch_loss.append(sum(batch_loss) / len(batch_loss))  # 计算并存储当前训练周期的平均损失。
        gradients = {name: param.grad.clone() for name, param in net.named_parameters()}

    return  net.state_dict(), sum(epoch_loss) / len(epoch_loss)




def train1(args, net, trainloader, optimizer):
    net.train()  # 将模型设置为训练模式
    # train and update
    # 定义随机梯度下降（SGD）优化器，用于更新模型参数。

    criterion = torch.nn.CrossEntropyLoss()

    epoch_loss = []  # 初始化一个列表，用于存储每个训练周期的损失
    gradient_list = []
    for iter in range(args.local_ep):  # 对每个训练周期进行迭代，5次#default 5 E
        batch_loss = []  # 初始化一个列表，用于存储每个批次的损失。
        for images, labels in trainloader:  # 对本地数据加载器中的每个批次进行迭代。
            images, labels = images.to(args.device), labels.to(args.device)  # 将图像和标签移动到设备上（通常是 GPU）
            images.requires_grad = True  # 确保输入数据可以计算梯度
            optimizer.zero_grad()  # 清零模型的梯度
            log_probs = net(images)  # 将图像输入模型，并获取预测的对数概率
            loss = criterion(log_probs, labels)  # 计算预测结果与真实标签之间的交叉熵损失。
            loss.backward()  # 反向传播，计算梯度
            gradients = {name:param.grad.clone() for name, param in net.named_parameters()}
            # clip gradient
            C = 1
            # Calculate the global norm of the gradients
            grad_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in gradients.values()))
            scaling=max(1, grad_norm / C)
            # Clip gradients and add Gaussian noise
            for name, grad in gradients.items():
                grad_ = grad / scaling # Clip the gradient
                noise = torch.normal(0, C, size=grad.size())
                gradients[name] = grad_ + noise
            optimizer.step()  # 更新模型参数
            batch_loss.append(loss.item())  # 将当前批次的损失值添加到 batch_loss 列表中
        epoch_loss.append(sum(batch_loss) / len(batch_loss))  # 计算并存储当前训练周期的平均损失。
        gradient_list.append(gradients)
    # Compute the average of gradients across all iterations
    avg_gradients = {}
    for name in gradient_list[0].keys():
        avg_gradients[name] = sum([grads[name] for grads in gradient_list]) / len(gradient_list)
    return avg_gradients, net.state_dict(), sum(epoch_loss) / len(epoch_loss)
