'''import copy
import torch
from torch import nn


import numpy as np
from collections import OrderedDict

# 将 OrderedDict 转换为 NumPy 数组
def dict_to_array(d):
    return np.array(list(d.values()))

# 计算两个向量之间的欧几里得距离的平方
def compute_distance_squared(v1, v2):
    return np.sum((v1 - v2)**2)

# Krum 函数实现了 Krum 聚合算法，它接收各客户端本地训练得到的模型参数，计算出最终的全局模型参数。
def Krum(V, f): 
    V = copy.deepcopy(V[0])
    n = V.keys() # 客户端数量
    scores = [] # 分数列表

    # 对于每个工作进程 i
    for i in n:
        score_i = 0 # 初始化分数为0
        distances = [] # 用于存储与其他工作进程的距离
        # 对于每个其他工作进程 j
        for j in n:
            if i != j: # 排除自己与自己的距离
                # 计算与其他工作进程的距离的平方，并将结果添加到距离列表中
                distance = compute_distance_squared(dict_to_array(V[i]), dict_to_array(V[j]))
                distances.append(distance)
        # 对距离列表进行排序，选择 n - f - 2 个最小的距离
        distances.sort()
        # 计算该工作进程的分数，即距离之和
        for k in range(n - f - 2):
            score_i += distances[k]
        # 添加分数到分数列表中
        scores.append(score_i)
    
    # 找到分数最小的工作进程 i
    min_index = np.argmin(scores)
    
    # 返回该工作进程的模型参数
    return V[min_index]'''
import copy
import torch
import numpy as np
from collections import OrderedDict

# 计算两个 OrderedDict 之间的欧几里得距离的平方
def compute_distance_squared(dict1, dict2):
    squared_sum = 0.0
    for key in dict1:
        squared_sum += torch.sum((dict1[key] - dict2[key])**2).item()
    return squared_sum

# Krum 聚合函数
def Krum(V, f):
    n = len(V)  # 客户端数量
    scores = []  # 分数列表

    # 对于每个工作进程 i
    for i in range(n):
        score_i = 0  # 初始化分数为 0
        distances = []  # 用于存储与其他工作进程的距离

        # 对于每个其他工作进程 j
        for j in range(n):
            if i != j:  # 排除自己与自己的距离
                # 计算与其他工作进程的距离的平方，并将结果添加到距离列表中
                distance = compute_distance_squared(V[i], V[j])
                distances.append(distance)
        
        # 对距离列表进行排序，选择 n - f - 2 个最小的距离 
        distances.sort()
        # 计算该工作进程的分数，即距离之和
        score_i = sum(distances[:n - f - 2])
        '''for k in range(n - f - 2):
               score_i += distances[k]'''
        
           
        
        # 添加分数到分数列表中
        scores.append(score_i)
    print(scores)
    # 找到分数最小的工作进程 i
    min_index = np.argmin(scores)
    print(min_index)
    
    # 返回该工作进程的模型参数
    return V[min_index], min_index


 