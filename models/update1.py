import copy

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



#这段代码定义了一个名为 DatasetSplit 的自定义数据集类，用于创建一个子数据集（用户的样本集），该子数据集包含原始数据集中特定索引集合的样本
class DatasetSplit(Dataset):#定义了一个名为 DatasetSplit 的类，该类继承自 PyTorch 的 Dataset 类，用于创建自定义数据集
    def __init__(self, dataset, idxs):#初始化方法，接受两个参数：dataset 是原始数据集对象，
        #idxs 是子数据集的索引集合，idxs=dict_users[idx]是第idx个用户被分配的图像索引的集合，{n0, n1, n2, n3, n4, ...}第n个用户。
        self.dataset = dataset
        self.idxs = list(idxs)#将索引集合转换为列表，并存储在实例变量 self.idxs 中。

    def __len__(self):
        return len(self.idxs)#返回子数据集索引集合的长度，即子数据集的样本数量，即单个用户的样本数量，假设为60个

    def __getitem__(self, item):#用于获取数据集中特定索引的样本，即获取单个用户的样本中某一个索引
        image, label = self.dataset[self.idxs[item]]#根据子数据集的索引集合中的第 item 个索引，从原始数据集中获取相应的图像和标签。
        return image, label


class LocalUpdate1(object):
    def __init__(self, args, dataset=None, idxs=None):
        #初始化方法，接受三个参数：args 是用于配置训练的参数对象，dataset 是原始数据集对象，idxs 是本地客户端的索引集合。
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()#定义交叉熵损失函数，用于计算损失
        self.selected_clients = []#初始化一个空列表 selected_clients，用于存储选择的客户端。
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.nk = len(idxs)
    def train(self, net):
        #net.train()#将模型设置为训练模式
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        #avg_gradients = {name: torch.zeros_like(param) for name, param in net.named_parameters()}
        epoch_loss = []  # 初始化一个列表，用于存储每个训练周期的损失

        gradient = {name: torch.zeros_like(param) for name, param in net.named_parameters()}
        #gradient = {name: torch.zeros_like(param) for name, param in net.state_dict().items()}
        for iter in range(self.args.local_ep):  # 对每个训练周期进行迭代，5次#default 5 E
            batch_loss = []  # 初始化一个列表，用于存储每个批次的损失。
            for batch_idx, (images, labels) in enumerate(self.ldr_train):  # 对本地数据加载器中的每个批次进行迭代。
                images, labels = images.to(self.args.device), labels.to(self.args.device)  # 将图像和标签移动到设备上（通常是 GPU）
                optimizer.zero_grad()  # 清零模型的梯度
                log_probs = net(images)  # 将图像输入模型，并获取预测的对数概率
                loss = self.loss_func(log_probs, labels)  # 计算预测结果与真实标签之间的交叉熵损失。
                loss.backward()  # 反向传播，计算梯度
                optimizer.step()  # 更新模型参数
                # Accumulate gradients
                for name, param in net.named_parameters():
                    gradient[name] += param.grad.clone().detach()
                batch_loss.append(loss.item())  # 将当前批次的损失值添加到 batch_loss 列表中
            epoch_loss.append(sum(batch_loss) / len(batch_loss))  # 计算并存储当前训练周期的平均损失。
        #gradient = {name: param.grad.clone() for name, param in net.named_parameters()}

        # Compute average gradients


        return sum(epoch_loss) / len(epoch_loss), gradient,self.nk


class LocalUpdate3(object):
    def __init__(self, args, dataset=None, idxs=None):
        #初始化方法，接受三个参数：args 是用于配置训练的参数对象，dataset 是原始数据集对象，idxs 是本地客户端的索引集合。
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()#定义交叉熵损失函数，用于计算损失
        self.selected_clients = []#初始化一个空列表 selected_clients，用于存储选择的客户端。
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.nk = len(idxs)
    def train(self, net):
        #net.train()#将模型设置为训练模式
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        #avg_gradients = {name: torch.zeros_like(param) for name, param in net.named_parameters()}
        epoch_loss = []  # 初始化一个列表，用于存储每个训练周期的损失
        #w_glob = copy.deepcopy(w_glob)
        gradient = {name: torch.zeros_like(param) for name, param in net.named_parameters()}
        #gradient = {name: torch.zeros_like(param) for name, param in net.state_dict().items()}
        for iter in range(self.args.local_ep):  # 对每个训练周期进行迭代，5次#default 5 E
            batch_loss = []  # 初始化一个列表，用于存储每个批次的损失。
            for batch_idx, (images, labels) in enumerate(self.ldr_train):  # 对本地数据加载器中的每个批次进行迭代。
                images, labels = images.to(self.args.device), labels.to(self.args.device)  # 将图像和标签移动到设备上（通常是 GPU）
                optimizer.zero_grad()  # 清零模型的梯度
                log_probs = net(images)  # 将图像输入模型，并获取预测的对数概率
                loss = self.loss_func(log_probs, labels)  # 计算预测结果与真实标签之间的交叉熵损失。
                loss.backward()  # 反向传播，计算梯度
                optimizer.step()  # 更新模型参数
                # Accumulate gradients
                for name, param in net.named_parameters():
                    gradient[name] += param.grad.clone().detach()

                ####LDP
                C = self.args.C  # Gradient norm bound
                sigma = self.args.sigma  # Noise scale
                grad_norm = 0.0
                # Iterate over each tensor in the state_dict
                for key, param in gradient.items():
                    grad_norm += torch.sum(param ** 2)
                # Take the square root of the sum of squared norms
                grad_norm = torch.sqrt(grad_norm)
                # Clip the gradient
                clipped = {name: torch.zeros_like(param) for name, param in gradient.items()}
                if grad_norm > C:
                    for key, param in gradient.items():
                        clipped[key] = param * (C / grad_norm)
                else:
                    for key, param in gradient.items():
                        clipped[key] = param
                # add noise
                grad_noisy = {name: param + torch.normal(mean=0, std=sigma, size=param.shape)
                                  for name, param in clipped.items()}


                batch_loss.append(loss.item())  # 将当前批次的损失值添加到 batch_loss 列表中
            epoch_loss.append(sum(batch_loss) / len(batch_loss))  # 计算并存储当前训练周期的平均损失。
        #gradient = {name: param.grad.clone() for name, param in net.named_parameters()}

        # Compute average gradients


        return sum(epoch_loss) / len(epoch_loss), grad_noisy,self.nk



class DatasetSplit(Dataset):#定义了一个名为 DatasetSplit 的类，该类继承自 PyTorch 的 Dataset 类，用于创建自定义数据集
    def __init__(self, dataset, idxs):#初始化方法，接受两个参数：dataset 是原始数据集对象，
        #idxs 是子数据集的索引集合，idxs=dict_users[idx]是第idx个用户被分配的图像索引的集合，{n0, n1, n2, n3, n4, ...}第n个用户。
        self.dataset = dataset
        self.idxs = list(idxs)#将索引集合转换为列表，并存储在实例变量 self.idxs 中。

    def __len__(self):
        return len(self.idxs)#返回子数据集索引集合的长度，即子数据集的样本数量，即单个用户的样本数量，假设为60个

    def __getitem__(self, item):#用于获取数据集中特定索引的样本，即获取单个用户的样本中某一个索引
        image, label = self.dataset[self.idxs[item]]#根据子数据集的索引集合中的第 item 个索引，从原始数据集中获取相应的图像和标签。
        return image, label


class LocalUpdate2(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.nk = len(idxs)  # The number of data points for this client


    def train(self, net):
        net.train()
        epoch_loss = []
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # Save the old state of the network (weights and biases only)
        w_old = {name: param.clone() for name, param in net.state_dict().items()}
        w = {name: param.clone() for name, param in net.named_parameters()}
        #w_old = copy.deepcopy(net.state_dict())

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        # Save the new state of the network (weights and biases only)
        w_new = {name: param.clone() for name, param in net.state_dict().items()}
        #w_new = net.state_dict()
        pseudo_gradients = {name: w_new[name] - w_old[name] for name in w_new.keys()}
        return w_new, sum(epoch_loss) / len(epoch_loss), pseudo_gradients, self.nk

class LocalUpdate4(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.nk = len(idxs)  # The number of data points for this client


    def train(self, net):
        net.train()
        epoch_loss = []
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # Save the old state of the network (weights and biases only)
        w_old = {name: param.clone() for name, param in net.state_dict().items()}

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)  # 将图像和标签移动到设备上（通常是 GPU）
                optimizer.zero_grad()  # 清零模型的梯度
                log_probs = net(images)  # 将图像输入模型，并获取预测的对数概率
                loss = self.loss_func(log_probs, labels)  # 计算预测结果与真实标签之间的交叉熵损失。
                loss.backward()  # 反向传播，计算梯度
                # Apply LDP
                C = self.args.C
                sigma = self.args.sigma
                # Calculate the norm of the gradients
                grad_norm = torch.tensor(0.0, device=self.args.device)
                for param in net.parameters():
                    if param.grad is not None:
                        grad_norm += torch.sum(param.grad ** 2)
                grad_norm = torch.sqrt(grad_norm)


                # Clip the gradients
                for param in net.parameters():
                    if param.grad is not None:
                        if grad_norm > C:
                            param.grad *= (C / grad_norm)
                        else:
                            param.grad *= 1

                # Add noise to the gradients
                for param in net.parameters():
                    if param.grad is not None:
                        noise = torch.normal(mean=0, std=sigma*C, size=param.grad.shape).to(self.args.device)
                        param.grad += noise

                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        # Save the new state of the network (weights and biases only)
        w_new = {name: param.clone() for name, param in net.state_dict().items()}
        #w_new = net.state_dict()
        gradient = {name: w_new[name] - w_old[name] for name in w_new.keys()}

        return w_new, sum(epoch_loss) / len(epoch_loss), gradient, self.nk




