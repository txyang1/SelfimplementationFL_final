import copy
import torch
from torch import nn

def FedAvge(w): # w：一个列表，包含各客户端本地训练得到的模型权重。每个元素都是一个 OrderedDict 字典，存储模型的参数。
    w_avg = copy.deepcopy(w[0]) # 深拷贝列表中的第一个权重字典，用于初始化全局平均权重 w_avg。
                                  # 深拷贝确保 w_avg 与 w[0] 相互独立，避免直接修改 w[0] 的内容
    
    mt = 0 # 初始化总的键数量为 0
    
    for k in range(1, len(w)): # 遍历各个客户端的权重
        nk = len(list(w[k].keys())) # 计算当前客户端权重的键数量
        mt += nk # 累加到总的键数量中
        
    for i in w_avg.keys():# 遍历权重字典 w_avg 中的每个键（即每个模型参数的名称）
        w_avg[i] = w_avg[i].float()
        for k in range(1, len(w)): # 从第一个客户端开始，累加其他客户端对应参数的权重到 w_avg 中
            nk = len(list(w[k].keys())) # 获取当前客户端权重的键数量
            w_avg[i] += torch.mul(w[k][i].float(), float(nk)) # 累加其他客户端对应参数的权重到 w_avg 中
        
        w_avg[i] = torch.div(w_avg[i], float(mt))
    
    return w_avg


def fedavg(w, nk_list):
    # w: a list of model weights from each client (as OrderedDicts)
    # nk_list: a list of the number of samples for each client

    # Deep copy the first client's weights as the initial averaged weights
    w_avg = copy.deepcopy(w[0])

    # Calculate the total number of samples across all clients (mt)
    mt = sum(nk_list)

    # Initialize w_avg with zeros
    w_avg = {key: torch.zeros_like(param) for key, param in w[0].items()}

    # Accumulate the weighted sum of model parameters from the other clients
    for i in w_avg.keys():
        for k in range(1, len(w)):
            w_avg[i] += w[k][i].float() * (nk_list[k] / mt)

    return w_avg