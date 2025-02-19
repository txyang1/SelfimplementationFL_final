import numpy as np
import numpy as np
import torch
from torch import nn, optim, hub
import torch.nn.functional as F

class ConvNet(nn.Module):

    def __init__(self, input_size=24, input_channel=3, classes=10, kernel_size=5, filters1=64, filters2=64, fc_size=384):
        """
            MNIST: input_size 28, input_channel 1, classes 10, kernel_size 3, filters1 30, filters2 30, fc200
            Fashion-MNIST: the same as mnist
            KATHER: input_size 150, input_channel 3, classes 8, kernel_size 3, filters1 30, filters2 30, fc 200
            CIFAR10: input_size 24, input_channel 3, classes 10, kernel_size 5, filters1 64, filters2 64, fc 384
        """
        super(ConvNet, self).__init__()
        self.input_size = input_size
        self.filters2 = filters2
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=filters1, kernel_size=kernel_size, stride=1, padding=padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=filters1, out_channels=filters2, kernel_size=kernel_size, stride=1, padding=padding)
        self.fc1 = nn.Linear(filters2 * input_size * input_size // 16, fc_size)
        self.fc2 = nn.Linear(fc_size, classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.filters2 * self.input_size * self.input_size // 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
def krum_(samples, f):
    size = len(samples)
    size_ = size - f - 2
    metric = []
    for idx in range(size):
        sample = samples[idx]
        samples_ = samples.copy()
        del samples_[idx]
        dis = np.array([np.linalg.norm(sample-sample_) for sample_ in samples_])
        metric.append(np.sum(dis[np.argsort(dis)[:size_]]))
    return metric

def krum1(samples, f):
    metric = krum_(samples, f)
    index = np.argmin(metric)
    return samples[index], index

def attack_xie(local_grads, weight, choices, mal_index):
    attack_vec = []
    for i, pp in enumerate(local_grads[0]):
        tmp = np.zeros_like(pp)
        for ji, j in enumerate(choices):
            if j not in mal_index:
                tmp += local_grads[j][i]
        attack_vec.append((-weight) * tmp / len(choices))
    for i in mal_index:
        local_grads[i] = attack_vec
    return local_grads


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision

MAL_FEATURE_TEMPLATE = '../data/%s_mal_feature_10.npy'
MAL_TARGET_TEMPLATE = '../data/%s_mal_target_10.npy'
MAL_TRUE_LABEL_TEMPLATE = '../data/%s_mal_true_label_10.npy'


class MalDataset(Dataset):
    def __init__(self, feature_path, true_label_path, target_path, transform=None):
        self.feature = np.load(feature_path)
        self.mal_dada = np.load(feature_path)
        self.true_label = np.load(true_label_path)
        self.target = np.load(target_path)

        self.transform = transform

    def __getitem__(self, idx):
        sample = self.feature[idx]
        mal_data = self.mal_dada[idx]
        if self.transform:
            sample = self.transform(sample)
            mal_data = self.transform(mal_data)
        return sample, mal_data, self.true_label[idx], self.target[idx]

    def __len__(self):
        return self.target.shape[0]
