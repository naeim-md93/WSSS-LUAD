import mxnet
import torch
from torch import nn

from src.models.resnet38d import ResNet38


class ResNet38ClassificationModel(ResNet38):
    def __init__(self, num_classes):
        super().__init__()

        del self.linear1000

        # -------- B7 -----------#
        self.dropout7 = nn.Dropout2d(p=0.5)
        # -----------------------#

        self.fc8 = nn.Conv2d(in_channels=4096, out_channels=num_classes, kernel_size=1, bias=False)

        nn.init.xavier_uniform_(tensor=self.fc8.weight)
        self.not_training = [self.conv1a, self.res2a, self.res2b1, self.res2b2]
        self.from_scratch_layers = [self.fc8]

    def extract_features(self, imgs):

        x = self.conv1a(imgs)  # (b, 64, 224, 224)

        # -------- B2 -----------#
        x, x1a = self.res2a(x=x)  # (b, 128, 112, 112), (b, 64, 224, 224)
        x, x2a = self.res2b1(x=x)  # (b, 128, 112, 112), (b, 128, 112, 112)
        x, x2b1 = self.res2b2(x=x)  # (b, 128, 112, 112), (b, 128, 112, 112)

        # -------- B3 -----------#
        x, x2b2 = self.res3a(x=x)  # (b, 256, 56, 56), (b, 128, 112, 112)
        x, x3a = self.res3b1(x=x)  # (b, 256, 56, 56), (b, 256, 56, 56)
        x, x3b1 = self.res3b2(x=x)  # (b, 256, 56, 56), (b, 256, 56, 56)

        # -------- B4 -----------#
        x, x3b2 = self.res4a(x=x)  # (b, 512, 28, 28), (b, 256, 56, 56)
        x, x4a = self.res4b1(x=x)  # (b, 512, 28, 28), (b, 512, 28, 28)
        x, x4b1 = self.res4b2(x=x)  # (b, 512, 28, 28), (b, 512, 28, 28)
        x, x4b2 = self.res4b3(x=x)  # (b, 512, 28, 28), (b, 512, 28, 28)
        x, x4b3 = self.res4b4(x=x)  # (b, 512, 28, 28), (b, 512, 28, 28)
        x, x4b4 = self.res4b5(x=x)  # (b, 512, 28, 28), (b, 512, 28, 28)

        # -------- B5 -----------#
        x, x4b5 = self.res5a(x=x)  # (b, 1024, 28, 28), (b, 512, 28, 28)
        x, x5a = self.res5b1(x=x)  # (b, 1024, 28, 28), (b, 1024, 28, 28)
        x, x5b1 = self.res5b2(x=x)  # (b, 1024, 28, 28), (b, 1024, 28, 28)

        # -------- B6 -----------#
        x, x5b2 = self.res6a(x=x)  # (b, 2048, 28, 28), (b, 1024, 28, 28)

        # -------- B7 -----------#
        x, x6a = self.res7a(x=x)  # (b, 4096, 28, 28), (b, 2048, 28, 28)
        x = self.bn7(x)  # (b, 4096, 28, 28)
        x = self.relu7(x)  # (b, 4096, 28, 28)
        # -----------------------#

        return {
            'x1a': x1a,

            'x2a': x2a,
            'x2b1': x2b1,
            'x2b2': x2b2,

            'x3a': x3a,
            'x3b1': x3b1,
            'x3b2': x3b2,

            'x4a': x4a,
            'x4b1': x4b1,
            'x4b2': x4b2,
            'x4b3': x4b3,
            'x4b4': x4b4,
            'x4b5': x4b5,

            'x5a': x5a,
            'x5b1': x5b1,
            'x5b2': x5b2,

            'x6a': x6a,

            'x7a': x
        }

    def forward(self, x):  # (b, 3, 224, 224)

        # -------- B7 -----------#
        x = self.extract_features(imgs=x)['x7a']  # (b, 4096, 28, 28)
        x = self.dropout7(x)  # (b, 4096, 28, 28)
        # -----------------------#

        x = self.avgpool(x)  # (b, 4096, 1, 1)
        x = self.fc8(x)  # (b, 4, 1, 1)

        return x  # (b, 4, 1, 1)

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups

    def train(self, mode=True):
        super(ResNet38ClassificationModel, self).train(mode)

        for layer in self.not_training:

            if isinstance(layer, nn.Conv2d):
                layer.weight.requires_grad = False
                if layer.bias is not None:
                    layer.bias.requires_grad = False

            elif isinstance(layer, nn.Module):
                for c in layer.children():
                    if hasattr(c, 'weight'):
                        c.weight.requires_grad = False
                    if hasattr(c, 'bias'):
                        if c.bias is not None:
                            c.bias.requires_grad = False

        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

    def load_weights(self, weights_path):
        print(f'{"#" * 10} Loading Weights {"#" * 10}')
        if weights_path is not None:
            if weights_path[-7:] == '.params':
                weights_dict = self.convert_mxnet_weights_to_torch(weights_path=weights_path)

                # Strict=False because of linear1000 and fc8
                self.load_state_dict(state_dict=weights_dict, strict=False)
                print('Initialize model with MXNet weights')

            elif weights_path[-4:] == '.pth':
                self.load_state_dict(state_dict=torch.load(f=weights_path, map_location='cpu'), strict=True)
                print('Initialize model with Pytorch weights')

            else:
                raise NotImplementedError('Invalid model weights')
        else:
            print('Initialize model with random weights')

    def convert_mxnet_weights_to_torch(self, weights_path):
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
