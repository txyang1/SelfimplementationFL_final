import torch

def get_model_parameters_vector(model):
    param_list = []
    for i, param in enumerate(model):
        param_list.append(param.data.view(-1))
    param_vector = torch.cat(param_list)
    return param_vector