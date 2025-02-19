import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random
from utils.sampling import mnist_iid, mnist_noniid
from utils.options import args_parser
from utils.preparedataset import  prepare_dataset
from models.FedAvge import FedAvge
from models.krum import Krum
from models.test import test_img
from models.average import Avg
from models.addnoise import add_gaussian_noise_to_ordereddict
from models.trainandtest import train, test
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
from models.Client import Client
from models.trainandtest import train, test

def Agg(net_glob,args,trainloaders,testloader):
    net_glob.train()
    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []#存储每一轮训练的平均损失
    test_accuracy =[]
    
    w_locals = [w_glob for i in range(args.num_users)]#初始化每个客户端的权重列表，所有客户端初始权重都设置为全局模型的权重。???
    for iter in range(args.epochs):#循环进行多个训练轮次 default=10 t= 1,2,3....
        loss_locals = []#存储每个客户端的局部损失
        if not args.all_clients:#如果不是所有客户端都参与聚合
            w_locals = []#重新初始化局部权重列表。
        m = max(int(args.frac * args.num_users), 1)#计算参与训练的客户端数量，确保至少有一个客户端。#frac =0.1 num_users=100
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)#从100用户随机选择10个参与训练的客户端索引。
        for k in idxs_users:#：遍历选定的客户端 k k=1,...,10
            #：初始化本地更新对象，传入参数、训练数据集和客户端索引
            trainloader=trainloaders[k]
            local = Client(args=args,trainloader=trainloader)
            #在客户端进行本地训练，返回更新后的权重和损失。
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            #idx=1-10, numuber of w1 = iter in update*总批次数in update*num_users in update, w_locals = [w1,...,w10]
            #否则，将更新后的权重添加到列表中。
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        '''# 选择三个随机元素
        selected_indices = random.sample(range(len(w_locals)), 3)
        # 对选定的元素添加高斯噪声
        for idx in selected_indices:
            w_locals[idx] = add_gaussian_noise_to_ordereddict(w_locals[idx])'''
        
        # if bayzantine worker is 3
        q=11
        epsilon = 10
        n=len(w_locals)
        if n>2*q:
         selected_indices = random.sample(range(len(w_locals)), q)
         non_selected_indices = [i for i in range(m) if i not in selected_indices]
         non_byzantine_w_locals = [w_locals[idx] for idx in non_selected_indices]
         w_correct_avg = Avg(non_byzantine_w_locals)
         # selected indices are replaced by arbitrary vector
         '''for idx in selected_indices:
            w_locals[idx] = - epsilon * w_correct_avg'''
         for idx in selected_indices:
            for key in w_locals[idx].keys():
             w_locals[idx][key] = - epsilon * w_correct_avg[key]
        else:
            print('q must less than m/2')
            
        # update global weights
        #w_glob = FedAvge(w_locals)
        #f=3
        w_glob = Krum(w_locals,q)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        _,acc_test = test(net_glob, testloader=testloader)
        test_accuracy.append(copy.deepcopy(acc_test))
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}, test accuracy {:.3f} '.format(iter, loss_avg, acc_test))
        loss_train.append(loss_avg)

    
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train,label = 'Krum with attack{}'.format(q))
    plt.ylabel('train Loss')
    plt.xlabel('Epoch')
    plt.title('Krum_train_loss_epsilon{}_{}_{}'.format(epsilon,args.dataset,args.iid))
    plt.legend()
    plt.savefig('./save/krum_train_loss_epsilon{}_{}_{}.png'.format(epsilon,args.dataset, args.model, args.iid))
    plt.figure()
    plt.plot(range(len(test_accuracy)), test_accuracy,label='Krum with attack{}'.format(q))
    plt.ylabel('test accuracy %')
    plt.xlabel('Epoch')
    plt.title('krum_test_accuracy_epsilon{}_{}_{}'.format(epsilon,args.dataset,args.iid))
    plt.legend()
    plt.savefig('./save/krum_test_accuracy_epsilon{}_{}_{}.png'.format(epsilon,args.dataset, args.model, args.iid))
    
    