import copy
import torch
from torch import nn

#FedAvg 函数实现了联邦平均算法，它接收各客户端本地训练得到的模型权重，计算其平均值，得到新的全局模型权重。
def FedAvg(w):#w：一个列表，包含各客户端本地训练得到的模型权重。每个元素都是一个 OrderedDict，存储模型的参数。
    #深拷贝列表中的第一个权重字典，用于初始化全局平均权重 w_avg。
    #深拷贝确保 w_avg 与 w[0] 相互独立，避免直接修改 w[0] 的内容
    
    w_avg = copy.deepcopy(w[0])#initialize,numuber of w1 = 本地iter*总批次数*num_users, w = [w1,...,w10]
    
    for k in w_avg.keys():#遍历权重字典 w_avg 中(即w[0])的每个键（即每个模型参数的名称）,k=1-number of w1=E*B*批量总数
        #从第一个客户端开始，累加其他客户端对应参数的权重到 w_avg 中。
        for i in range(1, len(w)):#len(w)=随机选中的10个本地客户端数
            '''累加各客户端权重：对每个键，从第一个客户端开始，对其他客户端对应参数的权重进行累加到w_avg中。
            这是通过一个循环从1到len(w)遍历列表中的每个元素，即每个客户端，然后将其对应参数的权重值加到w_avg中对应参数的权重值上。'''
            w_avg[k] += w[i][k]#w[1][k]+...+w[10][k]
        w_avg[k] = torch.div(w_avg[k], len(w))#计算平均值：累加完所有客户端的权重之后，将每个参数的权重值除以客户端数量，得到参数的平均值。
    return w_avg