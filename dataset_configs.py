import os


def get_LUAD_HistoSeg_configs(dataset_path):

    cfgs = {}

    # Train Configurations
    cfgs['train'] = {}
    cfgs['train']['images'] = {
        'dir_name': '',
        'num_items': 16678,
        'path': os.path.join(dataset_path, 'train'),
    }

    # Val Configurations
    cfgs['val'] = {}
    cfgs['val']['images'] = {
        'dir_name': 'img',
        'num_items': 300,
        'path': os.path.join(dataset_path, 'val')
    }
    cfgs['val']['masks'] = {
        'dir_name': 'mask',
        'num_items': 300,
        'path': os.path.join(dataset_path, 'val')
    }

    # Test Configurations
    cfgs['test'] = {}
    cfgs['test']['images'] = {
        'dir_name': 'img',
        'num_items': 307,
        'path': os.path.join(dataset_path, 'test')
    }
    cfgs['test']['masks'] = {
        'dir_name': 'mask',
        'num_items': 307,
        'path': os.path.join(dataset_path, 'test')
    }

    return cfgs
