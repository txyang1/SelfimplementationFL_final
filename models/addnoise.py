import copy
import torch
def add_gaussian_noise_to_ordereddict(odict, mean=0, stddev=1):
    noisy_odict = copy.deepcopy(odict)
    for key in noisy_odict.keys():
        if isinstance(noisy_odict[key], torch.Tensor):
            noise = torch.normal(mean=torch.zeros_like(noisy_odict[key]), std=stddev)
            noisy_odict[key] += noise
    return noisy_odict