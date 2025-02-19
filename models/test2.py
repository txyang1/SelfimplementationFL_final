import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

#定义了一个名为 test_img 的函数，接受三个参数：net_g 是要测试的模型，datatest 是测试数据集，args 是用于配置测试的参数对象。
def test_img(net_g, datatest, args):
    net_g.eval()#将模型设置为评估模式，这会关闭模型中的 dropout 和 batch normalization
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)#创建一个数据加载器，用于在测试过程中迭代加载测试数据集的批次。
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):#对数据加载器中的每个批次进行迭代，其中 data 是图像数据，target 是对应的标签。
        if args.gpu != -1:#如果指定了 GPU 设备
            data, target = data.cuda(), target.cuda()#将数据和目标标签移动到 GPU 上。
        log_probs = net_g(data)#将图像数据输入模型，并获取预测的对数概率
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()#计算当前批次的交叉熵损失，并累加到总的测试损失中。
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]#获取预测的类别标签
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()#统计预测正确的样本数量。

    test_loss /= len(data_loader.dataset)#计算测试集上的平均损失
    accuracy = 100.00 * correct / len(data_loader.dataset)#计算测试集上的准确率。
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

import torch
def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
