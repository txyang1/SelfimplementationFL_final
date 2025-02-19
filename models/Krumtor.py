from collections import OrderedDict
import numpy as np
import torch

def compute_distance_squared(dict1, dict2):
    squared_sum = 0.0
    for key in dict1:
        squared_sum += torch.sum((dict1[key] - dict2[key])**2).item()
    return squared_sum

# Krum 聚合函数
def KrumUV(V,U, upperboundKRU):
    n = len(V)
    f = len(U)
    scores = []  # 分数列表
    scores2 = []
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
        #对距离列表进行排序，选择 n - f - 2 个最小的距离 
        distances.sort()
        # 计算该工作进程的分数，即距离之和
        score_i = sum(distances[:n - f - 2])
        # 添加分数到分数列表中
        scores.append(score_i)
    
    
    score_i2 = upperboundKRU # 
    # 添加分数到分数列表中
    scores2.append(score_i2)
    '''for i in range(n-f):
        score_i2 = 0  # 初始化分数为 0
        distances2 = []  # 用于存储与其他工作进程的距离

        # 对于每个其他工作进程 j
        
               # 计算与其他工作进程的距离的平方，并将结果添加到距离列表中
        distance2 = compute_distance_squared(V[i], U[1])
        distance2 = 2*distance2
        distances2.append(distance2)
        
        # 对距离列表进行排序，选择 2 个最小的距离 
        distances2.sort()
        # 计算该工作进程的分数，即距离之和
        score_i2 = sum(distances2[:2])
        # 添加分数到分数列表中
        scores2.append(score_i2)'''
    '''for i in range(f):
        score_i2 = 0  # 初始化分数为 0
        distances2 = []  # 用于存储与其他工作进程的距离

        # 对于每个其他工作进程 j
        for j in range(2):
            if i != j:
               # 计算与其他工作进程的距离的平方，并将结果添加到距离列表中
               distance2 = compute_distance_squared(U[i], U[j])
               distances2.append(distance2)
        
        # 对距离列表进行排序，选择 2 个最小的距离 
        distances2.sort()
        # 计算该工作进程的分数，即距离之和
        score_i2 = sum(distances2[:2])
        # 添加分数到分数列表中
        scores2.append(score_i2)'''
    
    print(scores)
    
    # 找到分数最小的工作进程 i
    min_index = np.argmin(scores)
    min_index2 = np.argmin(scores2)
    print(min_index)
    if scores2[min_index2] <= scores[min_index]:
       
       print('successful')
       print(scores2)
       print(min_index2)
       # 返回该工作进程的模型参数
       return U[min_index2]
    else:
        print('failed')
        print(scores2)
        print(min_index2)
        return V[min_index]
