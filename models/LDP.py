import numpy as np

'''class LDPModule:
    def __init__(self, alpha, c, rho):
        self.alpha = alpha  # 隐私参数 alpha，用于控制隐私级别
        self.c = c  # 截断范围参数 c，用于限制梯度值的范围
        self.rho = rho  # 精度参数 rho，用于将梯度值转换为有限的整数空间
    
    def perturb(self, gradients):
        """
        使用 \(\alpha\)-CLDP-Fed 进行梯度扰动。
        该方法接收一个梯度向量，并返回一个扰动后的梯度向量。
        """
        def apply_ordinal_cldp(gradient):
            sensitivity = 1  # 假设灵敏度为1
            beta = self.alpha / (2 * sensitivity)  # 计算噪声的尺度参数 beta
            noise = np.random.laplace(0, 1 / beta)  # 生成拉普拉斯噪声
            perturbed_gradient = gradient + noise  # 对梯度添加噪声
            # 对添加噪声后的梯度进行截断，确保其在 [-c * 10^rho, c * 10^rho] 范围内
            clipped_gradient = np.clip(perturbed_gradient, -self.c * 10 ** self.rho, self.c * 10 ** self.rho)
            return clipped_gradient
        
        # 对每个梯度值应用 ordinal CLDP 并生成扰动后的梯度向量
        perturbed_gradients = np.array([apply_ordinal_cldp(g) for g in gradients])
        return perturbed_gradients'''
import numpy as np
import torch

class LDPModule:
    def __init__(self, alpha, c, rho):
        self.alpha = alpha
        self.c = c
        self.rho = rho
    
    def perturb(self, state_dict,selected_layer):
        """
        使用 \(\alpha\)-CLDP-Fed 进行梯度扰动。
        """
        def apply_ordinal_cldp(gradient):
            sensitivity = 1
            beta = self.alpha / (2 * sensitivity)
            noise = np.random.laplace(0, 1 / beta, size=gradient.shape)
            perturbed_gradient = gradient + noise
            clipped_gradient = np.clip(perturbed_gradient, -self.c * 10 ** self.rho, self.c * 10 ** self.rho)
            return clipped_gradient
        
        
    
        perturbed_state_dict = {}
        # 遍历模型的状态字典，仅对选择的层进行扰动
        for key, tensor in state_dict.items():
            if selected_layer in key:
                gradient = tensor.numpy()  # 将 PyTorch 张量转换为 NumPy 数组
                perturbed_gradient = apply_ordinal_cldp(gradient)  # 对梯度进行扰动
                perturbed_state_dict[key] = torch.tensor(perturbed_gradient)  # 将扰动后的梯度转换回 PyTorch 张量，并存储到字典中
            else:
                perturbed_state_dict[key] = tensor  # 未选择的层保持不变
        
        return perturbed_state_dict  # 返回扰动后的参数字典
        
'''class LDPModule:
    def __init__(self, alpha, c, rho):
        self.alpha = alpha  # LDP中的隐私参数
        self.c = c          # 用于剪切梯度的范围参数
        self.rho = rho      # 用于设定精度的参数

    def perturb(self, state_dict, selected_layer):
        """
        对传入的模型参数（梯度）进行扰动，保证局部差分隐私。
        仅对选择的层进行扰动。
        """
        def apply_ordinal_cldp(gradient):
            # 设置灵敏度，通常设为1
            sensitivity = 1
            
            # 计算拉普拉斯分布的尺度参数beta
            beta = self.alpha / (2 * sensitivity)
            
            # 生成与梯度同形状的拉普拉斯噪声
            noise = np.random.laplace(0, 1 / beta, size=gradient.shape)
            
            # 对梯度添加噪声
            perturbed_gradient = gradient + noise
            
            # 将添加噪声后的梯度进行剪切，以限制其值域
            clipped_gradient = np.clip(perturbed_gradient, -self.c * 10 ** self.rho, self.c * 10 ** self.rho)
            
            return clipped_gradient
        
        # 创建一个空字典用于存储扰动后的参数
        perturbed_state_dict = {}
        
        # 遍历模型的状态字典，仅对选择的层进行扰动
        for key, tensor in state_dict.items():
            if selected_layer in key:
                gradient = tensor.numpy()  # 将 PyTorch 张量转换为 NumPy 数组
                perturbed_gradient = apply_ordinal_cldp(gradient)  # 对梯度进行扰动
                perturbed_state_dict[key] = torch.tensor(perturbed_gradient)  # 将扰动后的梯度转换回 PyTorch 张量，并存储到字典中
            else:
                perturbed_state_dict[key] = tensor  # 未选择的层保持不变
        
        return perturbed_state_dict  # 返回扰动后的参数字典'''
    
