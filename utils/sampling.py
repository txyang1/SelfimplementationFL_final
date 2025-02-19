#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

#定义了一个函数 mnist_iid，该函数接受两个参数：dataset 是 MNIST 数据集对象，num_users 是要采样的用户数量。
def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #计算每个用户应该获取的图像数量，这里假设所有用户平均分配图像。
    num_items = int(len(dataset)/num_users)# num_users = 100 ,假设有6000张，平均每个分配60张，即num_item = 60
    #初始化一个空字典 dict_users 来存储每个用户的图像索引，同时创建一个包含所有图像索引的列表 all_idxs = 6000。
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):#一共进行100次分配，因为假设100个用户
        #从 all_idxs 中随机选择 num_items 个索引，并将它们存储在 dict_users[i] 中。使用 replace=False 参数确保选择的索引不会重复。
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))#每次从6000张中不重复随机抽取60张分给一个用户
        #从 all_idxs 中移除已经分配给当前用户的索引，以确保每个图像只分配给一个用户。
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 定义一些常量
    num_shards, num_imgs = 200, 300  # num_shards: 切片数，num_imgs: 每个切片中的图像数
    idx_shard = [i for i in range(num_shards)]  # 创建一个包含200个切片索引的列表
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  # 为每个用户创建一个空的索引数组字典
    idxs = np.arange(num_shards*num_imgs)  # 创建一个包含所有图像索引的数组
   
    labels = dataset.train_labels.numpy()  # 获取数据集中的标签，并转换为numpy数组

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 将图像索引和对应的标签堆叠成一个二维数组
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]  # 按标签排序
    idxs = idxs_labels[0,:]  # 获取排序后的图像索引

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))  # 随机选择2个切片
        idx_shard = list(set(idx_shard) - rand_set)  # 从切片列表中移除已分配的切片
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)  # 将选定的图像索引分配给用户
    return dict_users






