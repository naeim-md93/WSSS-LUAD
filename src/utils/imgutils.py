import os
import cv2
import torch
from skimage import morphology
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T


def imagenet_normalizer(images):
    normalizer = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return normalizer(images)


def imagenet_unnormalizer(images):
    unnormalizer = T.Compose([
        T.Normalize(mean=(0.0, 0.0, 0.0), std=(1/0.229, 1/0.224, 1/0.225)),
        T.Normalize(mean=(-0.485, -0.456, -0.406), std=(1.0, 1.0, 1.0))
    ])
    return unnormalizer(images)


def interpolate(x, s=224):
    x = F.interpolate(input=x, size=(s, s), mode='bilinear', align_corners=False)
    return x


def put_palette(x, bg):
    RED = [205, 51, 51]
    GREEN = [0, 255, 0]
    BLUE = [65, 105, 225]
    ORANGE = [255, 165, 0]
    WHITE = [255, 255, 255]

    x = x.numpy()
    x = np.concatenate((x, bg), axis=1)
    x = np.argmax(x, 1)[0]
    x = x.astype(np.uint8)
    x = Image.fromarray(obj=x, mode='P')
    x.putpalette(data=RED + GREEN + BLUE + ORANGE + WHITE)
    return x


def grad_cam(a, g):
    g = torch.mean(input=g, dim=(-2, -1), keepdim=True)
    a = g * a
    a = torch.sum(input=a, dim=1, keepdim=True)
    return a



def get_backgrounds(names, images, save_path):
    backgrounds = np.zeros(shape=(images.size(0), images.size(2), images.size(3)))

    for i in range(backgrounds.shape[0]):
        img = images[i].numpy()  # (c, h, w)
        img = np.moveaxis(a=img, source=0, destination=-1) * 255.0  # (h, w, c)

        img = img.astype(np.uint8)  # (h, w, c)
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2GRAY)  # (h, w)
        ret, binary = cv2.threshold(src=gray, thresh=200, maxval=255, type=cv2.THRESH_BINARY)  # (, ), (h, w)
        binary = np.uint8(binary)  # (h, w)
        dst = morphology.remove_small_objects(ar=(binary == 255), min_size=50, connectivity=1)  # (h, w)
        bg = dst * 1  # (h, w)

        if save_path is not None:
            cv2.imwrite(filename=os.path.join(save_path, names), img=(bg * 255.0))
        backgrounds[i] = bg

    return backgrounds  # (h, w)


class Grad_CAM:
    def __init__(self, model, target_layers, device, num_classes=4):
        self.target_layers = target_layers
        self.device = device
        self.num_classes = num_classes

        self.activations = {}
        self.gradients = {}

        self.model = model.to(device=device)

        for name, module in self.model.named_modules():
            if name in target_layers:
                module.register_forward_hook(hook=self.get_activations(name=name))
                module.register_full_backward_hook(hook=self.get_gradients(name=name))

    def get_activations(self, name):
        def hook(model, input, output):
            print(f'Act: {type(output), len(output)}')
            self.activations[name] = output.clone().detach().cpu()
        return hook

    def get_gradients(self, name):
        def hook(model, grad_input, grad_output):
            print(f'Grad: {type(grad_output),len(grad_output), type(grad_output[0]), len(grad_output[0])}')
            self.gradients[name] = grad_output[0].clone().detach().cpu()
        return hook

    def __call__(self, images, labels, rev_trans):

        cams = {l: {'acts': [], 'grads': []} for l in self.target_layers}

        for i in range(labels.size(1)):
            if labels[0, i] == 1:
                preds = self.model(images)
                preds = torch.relu(input=preds)

                tg = torch.zeros(size=(1, labels.size(1)))
                tg[0, i] = 1
                tg = tg.requires_grad_(True)
                tg = tg.to(device=self.device)
                tg = torch.sum(input=preds * tg)

                self.model.zero_grad()

                tg.backward()

                grads = list(self.gradients.values())  # [(1, c, h, w), ...]
                grads.reverse()
                grads = [grads[j].detach().cpu() for j in range(len(grads))]

                acts = list(self.activations.values())  # [(1, c, h, w), ...]
                acts = [acts[j].detach().cpu() for j in range(len(acts))]

                assert len(grads) == len(acts)

                for j in range(len(grads)):
                    for k, rfn in enumerate(rev_trans):
                        grads[j][k] = rfn(grads[j][k])
                        acts[j][k] = rfn(acts[j][k])

                for j, l in enumerate(self.target_layers):
                    cams[l]['acts'].append(acts[j])
                    cams[l]['grads'].append(grads[j])

            else:
                for tl in self.target_layers:
                    if (tl == 'b4_3') or (tl == 'b4_5'):
                        zero = torch.zeros(size=(len(rev_trans), 512, 28, 28))
                        cams[tl]['acts'].append(zero)
                        cams[tl]['grads'].append(zero)
                    elif tl == 'b5_2':
                        zero = torch.zeros(size=(len(rev_trans), 1024, 28, 28))
                        cams[tl]['acts'].append(zero)
                        cams[tl]['grads'].append(zero)
                    elif tl == 'b6':
                        zero = torch.zeros(size=(len(rev_trans), 2048, 28, 28))
                        cams[tl]['acts'].append(zero)
                        cams[tl]['grads'].append(zero)
                    elif tl == 'bn7':
                        zero = torch.zeros(size=(len(rev_trans), 4096, 28, 28))
                        cams[tl]['acts'].append(zero)
                        cams[tl]['grads'].append(zero)
                    else:
                        raise ValueError(f'Invalid Target Layer {tl}')

        for l, value in cams.items():
            cams[l]['acts'] = torch.stack(tensors=value['acts'], dim=1)
            cams[l]['grads'] = torch.stack(tensors=value['grads'], dim=1)

        return cams

