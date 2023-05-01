import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def check_LUAD_HistoSeg_dataset(dataset_configs):
    # Train, Val, Test
    for key, value in dataset_configs.items():

        # Image, Mask
        for k, v in value.items():
            dir_name = v['dir_name']
            num_items = v['num_items']
            path = v['path']

            dataset_path = os.sep.join(path.split(sep=os.sep)[:-1])
            if os.path.exists(os.path.join(dataset_path, 'training')):
                os.rename(src=os.path.join(dataset_path, 'training'), dst=os.path.join(dataset_path, 'train'))

            # Check if path exists:
            if os.path.exists(os.path.join(path, dir_name)):
                print(f'{key} {k} exists')

                for file_name in os.listdir(os.path.join(path, dir_name)):
                    if file_name.startswith('.'):
                        os.remove(path=os.path.join(path, dir_name, file_name))

                file_names = os.listdir(os.path.join(path, dir_name))
                if len(file_names) == num_items:
                    print(f'{key} {k} verified')

                else:
                    raise FileNotFoundError(f'Number of {key} {k} should be {num_items} but is {len(file_names)}')
            else:
                raise FileNotFoundError(f'{key} {k} not exists at {os.path.join(path, dir_name)}')


def split_classification_data(images_path, val):
    print(f'{"#" * 10} Splitting Data {"#" * 10}')

    image_names = os.listdir(path=images_path)
    random.shuffle(image_names)
    val_size = int(len(image_names) * val)

    # (train, val)
    data = (image_names[:-val_size], image_names[-val_size:])

    print(
        f'Train data: {len(data[0])} ({len(data[0]) / len(image_names)}), '
        f'Val data: {len(data[1])} ({len(data[1]) / len(image_names)})'
    )

    return data


class LUAD_HistoSeg_Classification_Dataset(Dataset):
    def __init__(self, images_path, image_names, trans):
        super(LUAD_HistoSeg_Classification_Dataset, self).__init__()
        self.images_path = images_path
        self.image_names = image_names
        self.trans = trans

    def get_labels(self, img_name):
        labels = np.array([int(img_name[-12]), int(img_name[-10]), int(img_name[-8]), int(img_name[-6])])
        return labels

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_path, img_name)

        img = Image.open(fp=img_path)
        img = img.convert(mode='RGB')

        label = self.get_labels(img_name=img_name)

        sample = {'name': img_name, 'img': img, 'label': label}

        if self.trans is not None:
            sample = self.trans(sample)

        return sample


def get_LUAD_HistoSeg_classification_dataloader(images_path, image_names, trans, batch_size, shuffle):

    dataloader = DataLoader(
        dataset=LUAD_HistoSeg_Classification_Dataset(
            images_path=images_path,
            image_names=image_names,
            trans=trans,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=10,
        pin_memory=True
    )

    return dataloader
