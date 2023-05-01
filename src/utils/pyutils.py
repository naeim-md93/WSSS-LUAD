import os
import json
import torch
import random
import numpy as np
import pickle as pk
import pandas as pd


def set_seed(seed):
    random.seed(a=seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def check_duplicate_name(save_path, save_name, file_type):
    exist_file = 1
    tmp = f'{save_name}.{file_type}'
    while os.path.exists(os.path.join(save_path, tmp)):
        tmp = f'{save_name}_{exist_file}.{file_type}'
        exist_file += 1
    return tmp


def save_file(data, save_path, save_name='', file_type='pk', replace=True):

    # Create save path directory if not exists
    os.makedirs(name=save_path, exist_ok=True)

    # check if there is a file in save_path with the same name as save_name,
    # if so, add _1, _2, ... to end of save_name
    if replace:
        save_name = f'{save_name}.{file_type}'
    else:
        save_name = check_duplicate_name(save_path=save_path, save_name=save_name, file_type=file_type)

    if file_type == 'json':
        with open(file=os.path.join(save_path, save_name), mode='w') as fp:
            json.dump(obj=data, fp=fp, indent=4)

    elif file_type in ['pickle', 'pk']:
        with open(file=os.path.join(save_path, save_name), mode='wb') as fp:
            pk.dump(obj=data, file=fp, protocol=pk.HIGHEST_PROTOCOL)

    elif file_type == 'csv':
        data.to_csv(
            path_or_buf=os.path.join(save_path, save_name),
            columns=data.columns,
            index=False
        )
    else:
        raise NotImplementedError(f'File type {file_type} not implemented')

    return os.path.join(save_path, save_name)


def load_file(path):
    extension = path.split(sep='.')[-1]

    if extension == 'json':
        with open(file=path, mode='r') as fp:
            data = json.load(fp=fp)

    elif extension in ['pickle', 'pk']:
        with open(file=path, mode='rb') as fp:
            data = pk.load(file=fp)

    else:
        with open(file=path, mode='rb') as fp:
            data = pk.load(file=fp)

    return data


