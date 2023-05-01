import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class Random90Rotation(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.random() > self.p:
            if 'img' in sample:
                sample['img'] = sample['img'].transpose(Image.ROTATE_90)

            if 'mask' in sample:
                masks = sample['mask']
                if isinstance(masks, list):
                    for i, mask in enumerate(masks):
                        masks[i] = mask.transpose(Image.ROTATE_90)
                else:
                    masks = masks.transpose(Image.ROTATE_90)

                sample['mask'] = masks

        return sample


class Random180Rotation(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.random() > self.p:
            if 'img' in sample:
                sample['img'] = sample['img'].transpose(Image.ROTATE_180)

            if 'mask' in sample:
                masks = sample['mask']
                if isinstance(masks, list):
                    for i, mask in enumerate(masks):
                        masks[i] = mask.transpose(Image.ROTATE_180)
                else:
                    masks = masks.transpose(Image.ROTATE_180)

                sample['mask'] = masks

        return sample


class Random270Rotation(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.random() > self.p:
            if 'img' in sample:
                sample['img'] = sample['img'].transpose(Image.ROTATE_270)

            if 'mask' in sample:
                masks = sample['mask']
                if isinstance(masks, list):
                    for i, mask in enumerate(masks):
                        masks[i] = mask.transpose(Image.ROTATE_270)
                else:
                    masks = masks.transpose(Image.ROTATE_270)

                sample['mask'] = masks

        return sample


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.random() > self.p:
            if 'img' in sample:
                sample['img'] = sample['img'].transpose(Image.FLIP_LEFT_RIGHT)

            if 'mask' in sample:
                masks = sample['mask']

                if isinstance(masks, list):
                    for i, mask in enumerate(masks):
                        masks[i] = mask.transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    masks = masks.transpose(Image.FLIP_LEFT_RIGHT)

                sample['mask'] = masks

        return sample


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.random() > self.p:
            if 'img' in sample:
                sample['img'] = sample['img'].transpose(Image.FLIP_TOP_BOTTOM)

            if 'mask' in sample:
                masks = sample['mask']

                if isinstance(masks, list):
                    for i, mask in enumerate(masks):
                        masks[i] = mask.transpose(Image.FLIP_TOP_BOTTOM)
                else:
                    masks = masks.transpose(Image.FLIP_TOP_BOTTOM)

                sample['mask'] = masks

        return sample


class ImageToPatches(object):
    def __init__(self, patch_size, batch_first=False):
        self.ps = patch_size
        self.batch_first = batch_first

    def __call__(self, sample):
        if 'img' in sample:

            if self.batch_first:
                sample['img'] = sample['img'] \
                    .unfold(1, 3, 3) \
                    .unfold(2, self.ps[0], self.ps[1]) \
                    .unfold(3, self.ps[0], self.ps[1]) \
                    .reshape(-1, 3, self.ps[0], self.ps[1])
            else:
                raise NotImplementedError(f'Not implemented yet!')
                # sample['img'] = sample['img'] \
                #     .unfold(0, 3, 3) \
                #     .unfold(1, self.patch_size[0], self.patch_size[1]) \
                #     .unfold(2, self.patch_size[0], self.patch_size[1])
                # print(sample['img'].shape)
                    # .reshape(-1, 3, self.patch_size[0], self.patch_size[1])
        return sample


class PatchesToImage(object):
    def __init__(self, original_size, batch_first=False):
        self.os = original_size
        self.batch_first = batch_first

    def __call__(self, sample):
        if 'img' in sample:
            if self.batch_first:
                i = sample['img']
                i = i \
                    .reshape(-1, 1, self.os[0] // i.size(2), self.os[1] // i.size(3), i.size(1), i.size(2), i.size(3)) \
                    .permute(0, 1, 4, 2, 5, 3, 6) \
                    .contiguous() \
                    .view(-1, 3, self.os[0], self.os[1])
            else:
                raise NotImplementedError('Not implemented yet!')
                # sample['img'] = sample['img'] \
                #     .permute(0, 3, 1, 4, 2, 5) \
                #     .contiguous() \
                #     .view(3, self.original_size[0], self.original_size[1])
            sample['img'] = i
        return sample


class ToTensor(object):
    def __call__(self, sample):

        if 'img' in sample:
            sample['img'] = F.to_tensor(pic=sample['img'])

        if 'label' in sample:
            sample['label'] = torch.tensor(data=sample['label'], dtype=torch.long)

        if 'mask' in sample:
            masks = sample['mask']
            if isinstance(masks, list):
                for i, mask in enumerate(masks):
                    masks[i] = torch.tensor(data=np.array(mask), dtype=torch.long)
            else:
                masks = torch.tensor(data=np.array(masks), dtype=torch.long)

            sample['mask'] = masks

        return sample


# class RandomGaussianBlur(object):
#     def __init__(self, p):
#         self.p = p
#
#     def __call__(self, sample):
#
#         if 'img' in sample:
#             img = sample['img']
#             if np.random.random() < self.p:
#                 img = img.filter(ImageFilter.GaussianBlur(radius=np.random.random()))
#             sample['img'] = img
#
#         return sample


class RandomRemovePatch(object):
    def __init__(self, p, num_patches, img_size, patch_size, batch=False):
        self.p = p
        self.num_patches = num_patches
        self.img_size = img_size
        self.patch_size = patch_size
        self.batch = batch

    def __call__(self, sample):
        if 'img' in sample:
            img = sample['img']

            if np.random.random() < self.p:

                if self.batch:
                    for b in range(img.size(0)):
                        for _ in range(self.num_patches):
                            x = np.random.randint(low=0, high=self.img_size[0] - self.patch_size[0])
                            y = np.random.randint(low=0, high=self.img_size[1] - self.patch_size[1])

                            img[b, :, x: x + self.patch_size[0], y: y + self.patch_size[1]] = 1

                else:
                    for _ in range(self.num_patches):
                        x = np.random.randint(low=0, high=self.img_size[0] - self.patch_size[0])
                        y = np.random.randint(low=0, high=self.img_size[1] - self.patch_size[1])

                        img[:, x: x + self.patch_size[0], y: y + self.patch_size[1]] = 1

            sample['img'] = img

        return sample


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        if 'img' in sample:
            sample['img'] = F.normalize(tensor=sample['img'], mean=self.mean, std=self.std)

        return sample
