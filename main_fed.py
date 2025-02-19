#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.FedAvge import FedAvge
from models.krum import Krum
from models.test import test_img
from torch.utils.data import random_split


if __name__ == '__main__':
    # parse args
    args = args_parser()#这一行调用args_parser函数，解析命令行参数并返回一个包含参数的对象args。
                        #这些参数通常包括实验配置，比如数据集类型、模型类型、学习率、批次大小等。
                        #这一行代码设置计算设备为GPU或CPU。
                        #torch.cuda.is_available(): 检查是否有可用的CUDA GPU。
                        #args.gpu != -1: 检查命令行参数中是否指定了GPU编号。如果args.gpu不等于-1，表示用户希望使用特定的GPU。
        #torch.device('cuda:{}'.format(args.gpu)): 如果上述条件都满足，则创建一个指定GPU设备的torch.device对象，例如cuda:0表示使用第一块GPU。
                        #else 'cpu': 如果条件不满足，则使用CPU作为计算设备
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':#根据命令行参数args.dataset，加载相应的数据集（MNIST或CIFAR-10）并对用户进行数据划分。
        # 定义一个数据预处理和归一化的转换，其中ToTensor将图像转换为张量，Normalize标准化图像。
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)#下载并加载训练集，并应用预处理转换
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        
        # construct histogram
        all_labels = dataset_train.targets
        num_possible_labels = len(set(all_labels.numpy().tolist()))  # this counts unique labels (so it should be = 10)
        plt.hist(all_labels, bins=num_possible_labels)

        # construct histogram
        all_labels = dataset_test.targets
        num_possible_labels = len(set(all_labels.numpy().tolist()))  # this counts unique labels (so it should be = 10)
        plt.hist(all_labels, bins=num_possible_labels)

        # plot formatting
        plt.xticks(range(num_possible_labels))
        plt.grid()
        plt.xlabel("Label")
        plt.ylabel("Number of images")
        plt.title("Class labels distribution for MNIST")
        plt.savefig('./save/labels_distribution.png')


       


        # sample users
        if args.iid:#如果参数指定IID（独立同分布），则调用mnist_iid函数对训练数据进行IID划分。
            dict_users = mnist_iid(dataset_train, args.num_users)# num_users = 100， dataset_train 假设是6000 则dict_users=60
            print("Data is iid" )
            
        else:#否则调用mnist_noniid函数对训练数据进行非IID划分。
            dict_users = mnist_noniid(dataset_train, args.num_users)
            print("Data is noniid" )
    else:
        exit('Error: unrecognized dataset')
    
    
    '''trainloaders, valloaders, testloader = prepare_dataset(
    num_partitions=100, batch_size=32
)

# first partition
train_partition = trainloaders[0].dataset

# count data points
partition_indices = train_partition.indices
print(f"number of images: {len(partition_indices)}")

# visualise histogram
plt.hist(train_partition.dataset.dataset.targets[partition_indices], bins=10)
plt.grid()
plt.xticks(range(10))
plt.xlabel("Label")
plt.ylabel("Number of images")
plt.title("Class labels distribution for MNIST")'''
    #获取训练集中第一个样本的图像尺寸，这个信息在后续构建模型时会用到。
    img_size = dataset_train[0][0].shape

    # build model
    #根据命令行参数args.model和args.dataset构建相应的模型，并将其加载到指定的设备（GPU或CPU）。
    
    if args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)#创建一个CNNMnist模型实例，并将其移动到指定的设备。
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    #将模型设置为训练模式，这会启用诸如dropout等仅在训练期间启用的层。
    net_glob.train()

    # copy weights
    #state_dict是一个包含模型所有参数和缓存信息的字典（OrderedDict），
    #它包含了模型的所有可学习参数（例如权重和偏置）以及一些缓冲区（例如批归一化层的均值和方差）
    #通过调用net_glob.state_dict()，我们获取到当前全局模型net_glob的所有参数
    w_glob = net_glob.state_dict()

    # training
    loss_train = []#存储每一轮训练的平均损失
    test_accuracy =[]

    if args.all_clients: #如果所有客户端都参与聚合
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]#初始化每个客户端的权重列表，所有客户端初始权重都设置为全局模型的权重。???
    for iter in range(args.epochs):#循环进行多个训练轮次 default=10 t= 1,2,3....
        loss_locals = []#存储每个客户端的局部损失
        
        if not args.all_clients:#如果不是所有客户端都参与聚合
            w_locals = []#重新初始化局部权重列表。
        m = max(int(args.frac * args.num_users), 1)#计算参与训练的客户端数量，确保至少有一个客户端。#frac =0.1 num_users=100
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)#从100用户随机选择10个参与训练的客户端索引。
        for idx in idxs_users:#：遍历选定的客户端 k k=1,...,10
            #：初始化本地更新对象，传入参数、训练数据集和客户端索引
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])#dict_users是一个字典，idxs=dict_users[idx]是第idx个用户被分配的图像索引的集合
            #在客户端进行本地训练，返回更新后的权重和损失。
            w, loss,_ = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:#如果所有客户端都参与，将更新后的权重存入对应位置。
                w_locals[idx] = copy.deepcopy(w)#idx=1-10, numuber of w1 = iter in update*总批次数in update*num_users in update, w_locals = [w1,...,w10]
            else:#否则，将更新后的权重添加到列表中。
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        #这个函数接收所有客户端本地训练后的权重列表 w_locals，并计算其平均值，作为新的全局权重 w_glob
        w_glob = FedAvge(w_locals)

        
       
        
        # copy weight to net_glob
        #将更新后的全局权重 w_glob 复制到全局模型 net_glob 中。
        #load_state_dict 是 PyTorch 中的一个方法，用于加载模型的权重。
        net_glob.load_state_dict(w_glob)
        
        acc_test, _ = test_img(net_glob, dataset_test, args)
        test_accuracy.append(copy.deepcopy(acc_test))
        # print loss
        #计算当前训练轮次中所有客户端的平均损失。
        #loss_locals 是一个列表，存储了每个客户端本地训练的损失值。
        #通过求和并除以客户端数量，得到平均损失 loss_avg。
        loss_avg = sum(loss_locals) / len(loss_locals)
        #打印当前轮次 iter 的平均损失 loss_avg。
        #{:3d} 表示整数格式，宽度为3。
        #{:.3f} 表示浮点数格式，保留三位小数。
        print('Round {:3d}, Average loss {:.3f}, test accuracy {:.3f}'.format(iter, loss_avg, acc_test))
        #将当前轮次的平均损失 loss_avg 添加到 loss_train 列表中，以便后续分析和绘图。
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    
    plt.plot(range(len(loss_train)), loss_train)#使用 plt.plot() 函数绘制损失曲线。
                                                #range(len(loss_train)) 用作 x 轴，表示训练轮次；loss_train 是训练损失值，用作 y 轴。
    plt.ylabel('train_loss')
    #保存绘制的图形为文件
    #文件路径和名称的格式化字符串。
    #args.dataset、args.model、args.epochs、args.frac、args.iid 分别表示数据集名称、模型名称、轮次数、客户端参与比例和是否为IID分布。
    #这些参数会被插入到字符串中形成文件名，以便区分不同的实验结果。
    #plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    plt.savefig('./save/fed_train_loss')
    plt.figure()
    plt.plot(range(len(test_accuracy)), test_accuracy)
    plt.ylabel('test_accuracy')
    plt.savefig('./save/fed_test_accuracy')
    # testing
    net_glob.eval()#将全局模型 net_glob 设置为评估模式。在评估模式下，模型不会更新权重，也不会影响输入数据的梯度计算，因此可以加速推理过程
    ##调用 test_img 函数对全局模型在训练集上进行测试。该函数返回训练集上的准确率 acc_train 和损失 loss_train。
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    #调用 test_img 函数对全局模型在测试集上进行测试。该函数返回测试集上的准确率 acc_test 和损失 loss_test
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

