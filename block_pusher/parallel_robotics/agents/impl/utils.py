import math
import torch
import imgaug as ia
from imgaug import augmenters as iaa
''' 
    various utils
'''

augs = [
    iaa.LinearContrast((0.85,1.15), per_channel=0.25),
    iaa.Add((-10,10), per_channel=True),
    iaa.GammaContrast((0.90, 1.10)),
    iaa.GaussianBlur(sigma=(0.0,0.3)),
    iaa.MultiplySaturation((0.95,1.05)),
    iaa.AdditiveGaussianNoise(scale=(0,0.0125*255))
]
seq_augs = iaa.Sequential(augs, random_order=True)

def augment(image):
    return seq_augs(image=image).copy()

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

'''
    Update target networks for SAC
'''

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

'''
    Utility to request a required argument in a dotmap
'''
def get_required_argument(dotmap, key, message, default=None):
    val = dotmap.get(key, default)
    if val is default:
        raise ValueError(message)
    return val

