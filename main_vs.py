import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import csv
import numpy as np
from torchvision import datasets, transforms
import torch
import random
from utils.sampling import mnist_iid, mnist_noniid
from utils.options import args_parser
from utils.preparedataset import prepare_dataset
from models.update1 import LocalUpdate1
from models.Nets import MLP, CNNMnist, CNNCifar, CIFAR10_CNN
from models.FedAvge import FedAvge,fedavg
from models.krum import Krum
from models.test import test_img
from models.average import Avg
from models.Update import LocalUpdate
from utils.sampling import mnist_iid, mnist_noniid
from models.addnoise import add_gaussian_noise_to_ordereddict
from models.trainandtest import train, test
from models.Aggregation import Agg
from models.Client import Client
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
from models.checkiid import check_iid
from models.clienttrain import train1,train2
from models.Krumtor import compute_distance_squared, KrumUV
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
import torch
from torch import nn

from utils.sampling import mnist_iid, mnist_noniid
from utils.options import args_parser
from models.update1 import LocalUpdate2
from models.Nets import MLP, CNNMnist, CNNCifar
from models.FedAvge import FedAvge
from models.krum import Krum
from models.test import test_img
from torch.utils.data import random_split

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                       transform=trans_mnist)  # 下载并加载训练集，并应用预处理转换
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        train_size = 50000
        dataset_train.data = dataset_train.data[:train_size]
        dataset_train.targets = dataset_train.targets[:train_size]
        test_size = 10000
        dataset_test.data = dataset_test.data[:test_size]
        dataset_test.targets = dataset_test.targets[:test_size]
        trainloaders, valloaders, testloader = prepare_dataset(dataset_train, dataset_test, args.num_users,
                                                               batch_size=128)
        # sample users
        dict_users = mnist_iid(dataset_train,
                               args.num_users)
        '''if args.iid:  # 如果参数指定IID（独立同分布），则调用mnist_iid函数对训练数据进行IID划分。
            dict_users = mnist_iid(dataset_train,
                                   args.num_users)  # num_users = 100， dataset_train 假设是6000 则dict_users=60
            print("Data is iid")

        else:  # 否则调用mnist_noniid函数对训练数据进行非IID划分。
            dict_users = mnist_noniid(dataset_train, args.num_users)
            print("Data is noniid")'''
    elif args.dataset == 'cifar':

        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # 指定训练集和测试集的大小
        train_size = 50000
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_train.data = dataset_train.data[:train_size]
        dataset_train.targets = dataset_train.targets[:train_size]
        test_size = 10000
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        dataset_test.data = dataset_test.data[:test_size]
        dataset_test.targets = dataset_test.targets[:test_size]
        #trainloaders, valloaders, testloader = prepare_dataset(dataset_train, dataset_test, args.num_users,batch_size=128)
        # sample users

        dict_users = mnist_iid(dataset_train,args.num_users)  # num_users = 100， dataset_train 假设是6000 则dict_users=60



    else:
        exit('Error: unrecognized dataset')
        # sample users

    #img_size = dataset_train[0][0].shape

    # build model krum without attack ---------------------------------------------------------------------------------
    if args.model == 'cnn' and args.dataset == 'cifar' :
        net_glob = CIFAR10_CNN(args=args).to(args.device)
    else:
        net_glob = CNNMnist(args=args).to(args.device)

    print(net_glob)
    net_glob.train()
    # initial glob weights
    w_glob = net_glob.state_dict()
    # training
    loss_train = []  # 存储每一轮训练的平均损失
    test_accuracy = []
    selected_indices_list = []
    min_index_list = []

    #w_locals = [w_glob for i in range(args.num_users)]  # 初始化每个客户端的权重列表，所有客户端初始权重都设置为全局模型的权重。???
    for iter in range(args.epochs):  # 循环进行多个训练轮次 default=10 t= 1,2,3....
        loss_locals = []  # 存储每个客户端的局部损失
        nk_list = []
        if not args.all_clients:  # 如果不是所有客户端都参与聚合
            w_locals = []  # 重新初始化局部权重列表。
        m = max(int(args.frac * args.num_users), 1)  # 计算参与训练的客户端数量，确保至少有一个客户端。#frac =0.1 num_users=100
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # 从100用户随机选择10个参与训练的客户端索引。
        for idx in idxs_users:#：遍历选定的客户端 k k=1,...,10
            #：初始化本地更新对象，传入参数、训练数据集和客户端索引
            local = LocalUpdate2(args=args, dataset=dataset_train, idxs=dict_users[idx])#dict_users是一个字典，idxs=dict_users[idx]是第idx个用户被分配的图像索引的集合
            #在客户端进行本地训练，返回更新后的权重和损失。
            w, loss,_,nk = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:#如果所有客户端都参与，将更新后的权重存入对应位置。
                w_locals[idx] = copy.deepcopy(w)#idx=1-10, numuber of w1 = iter in update*总批次数in update*num_users in update, w_locals = [w1,...,w10]
            else:#否则，将更新后的权重添加到列表中。
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            nk_list.append(copy.deepcopy(nk))


        if args.attack == 'xie':
            if iter >= args.addtime:
                n = len(w_locals)
                if n > 2 * args.malicious+2:
                    selected_indices = random.sample(range(len(w_locals)), args.malicious)
                    print(selected_indices)
                    non_selected_indices = [i for i in range(m) if i not in selected_indices]
                    non_byzantine_w_locals = [w_locals[idx] for idx in non_selected_indices]
                    w_correct_avg = Avg(non_byzantine_w_locals)
                    for key in w_correct_avg.keys():
                        w_correct_avg[key] = -args.epsilon * w_correct_avg[key]
                    for idx in selected_indices:
                        w_locals[idx] = w_correct_avg
                else:
                    print('wrong, n must larger than 2q+2')
        #if iter >= args.addtime:
        #torch.save(w_locals, 'save/w_locals_epoch_5_att_epsilon1_378.pth')
        if args.Agg == 'Krum':
            w_glob,min_index = Krum(w_locals, args.malicious)
        elif args.Agg == 'Fedavg':
            w_glob = fedavg(w_locals, nk_list)

        # update the w_glob
        net_glob.load_state_dict(w_glob)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        #_, acc_test = test(net_glob, testloader=DataLoader(dataset_test, batch_size=128))
        test_accuracy.append(copy.deepcopy(acc_test))
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}, test accuracy {:.3f} '.format(iter, loss_avg, acc_test))
        loss_train.append(loss_avg)

        if iter >= args.addtime and args.Agg =='Krum':
            selected_indices_list.append(selected_indices)
            min_index_list.append(min_index)

    with open('./result/Instancenorm_{}_{}_attack_{}_epsilon_{}_addtime_{}_malicious_{}_epoch_{}_users_{}.txt'.format(
            args.dataset, args.Agg, args.attack, args.epsilon, args.addtime, args.malicious, args.epochs, args.frac),
              'w') as f:
        f.write("Epoch\tTest Accuracy\tMin Index\tSelected Indices\n")
        for epoch in range(args.epochs):
            if epoch >= args.addtime and args.Agg == 'Krum':
                f.write("{:d}\t{:.3f}\t{:.3f}\t{:d}\t{}\n".format(epoch, test_accuracy[epoch], loss_train[epoch],
                                                                  min_index_list[epoch - args.addtime], ','.join(
                        map(str, selected_indices_list[epoch - args.addtime]))))
            else:
                f.write("{:d}\t{:.3f}\t{:.3f}\tNA\tNA\n".format(epoch, test_accuracy[epoch], loss_train[epoch]))
