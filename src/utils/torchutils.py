import mxnet
import torch
import numpy as np
import torchvision.transforms as T


def min_max_scaler(x, dims, eps=1e-9):
    maxs = torch.amax(input=x, dim=dims)
    mins = torch.amin(input=x, dim=dims)
    x = (x - mins) / (maxs - mins + eps)
    return x, mins, maxs


def rev_min_max_scaler(x, mins, maxs, eps=1e-9):
    return x * (maxs - mins + eps) + mins

# def convert_mxnet_to_torch(filename):
#
#     save_dict = mxnet.nd.load(filename)
#
#     renamed_dict = dict()
#
#     bn_param_mx_pt = {'beta': 'bias', 'gamma': 'weight', 'mean': 'running_mean', 'var': 'running_var'}
#
#     for k, v in save_dict.items():
#
#         v = torch.from_numpy(v.asnumpy())
#         toks = k.split('_')
#
#         if 'conv1a' in toks[0]:
#             renamed_dict['conv1a.weight'] = v
#
#         elif 'linear1000' in toks[0]:
#             pass
#
#         elif 'branch' in toks[1]:
#
#             pt_name = []
#
#             if toks[0][-1] != 'a':
#                 pt_name.append('b' + toks[0][-3] + '_' + toks[0][-1])
#             else:
#                 pt_name.append('b' + toks[0][-2])
#
#             if 'res' in toks[0]:
#                 layer_type = 'conv'
#                 last_name = 'weight'
#
#             else:  # 'bn' in toks[0]:
#                 layer_type = 'bn'
#                 last_name = bn_param_mx_pt[toks[-1]]
#
#             pt_name.append(layer_type + '_' + toks[1])
#
#             pt_name.append(last_name)
#
#             torch_name = '.'.join(pt_name)
#             renamed_dict[torch_name] = v
#
#         else:
#             last_name = bn_param_mx_pt[toks[-1]]
#             renamed_dict['bn7.' + last_name] = v
#
#     return renamed_dict


def convert_mxnet_weights_to_torch(weights_path):
    def fix_name(n):
        n = n.replace('aux:', '').replace('arg:', '')
        tmp = n.split(sep='_')

        if n.startswith('conv1a'):
            n = f'{tmp[0]}.{tmp[1]}'

        elif n.startswith('res'):
            n = f'{tmp[0]}.conv_{tmp[1]}.{tmp[2]}'

        elif n.startswith('linear1000'):
            n = f'{tmp[0]}.{tmp[1]}'

        elif n.startswith('bn'):
            if tmp[0] != 'bn7':
                n = f'{tmp[0].replace("bn", "res")}.bn_{tmp[1]}_{"_".join(tmp[2:])}'
            n = n.replace('_beta', '.bias')
            n = n.replace('_gamma', '.weight')
            n = n.replace('_moving_mean', '.running_mean')
            n = n.replace('_moving_var', '.running_var')
        return n

    def fix_weight(w):
        return torch.from_numpy(w.asnumpy())

    mxnet_weights = mxnet.nd.load(fname=weights_path)
    torch_weights = {fix_name(n=n): fix_weight(w=w) for n, w in mxnet_weights.items()}
    return torch_weights


class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super(PolyOptimizer, self).__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super(PolyOptimizer, self).step(closure)

        self.global_step += 1


def standard_scale(x, eps=1e-9, dims=-2, numpy=False):
    if dims == -2:
        if numpy:
            mins = np.amin(a=x, axis=(-2, -1), keepdims=True)
            maxs = np.amax(a=x, axis=(-2, -1), keepdims=True)
        else:
            mins = torch.amin(input=x, dim=(-2, -1), keepdim=True)
            maxs = torch.amax(input=x, dim=(-2, -1), keepdim=True)
    elif dims == -3:
        if numpy:
            mins = np.amin(a=x, axis=(-3, -2, -1), keepdims=True)
            maxs = np.amax(a=x, axis=(-3, -2, -1), keepdims=True)
        else:
            mins = torch.amin(input=x, dim=(-3, -2, -1), keepdim=True)
            maxs = torch.amax(input=x, dim=(-3, -2, -1), keepdim=True)
    x = (x - mins) / (maxs - mins + eps)
    return x


def print_incorrects(names, probs, labels, thresh=0.5):
    incorrects = ((probs >= thresh) != labels).sum(axis=1) > 0
    for iy in range(len(incorrects)):
        if incorrects[iy] == 1:
            print(names[iy], probs[iy], labels[iy])
