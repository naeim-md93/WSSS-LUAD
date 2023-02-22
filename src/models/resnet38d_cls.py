import torch
from torch import nn
import torch.nn.functional as F

from src.utils import torchutils
from src.models.utils import ResBlock, ResBlockBottleneck


class ResNet38ClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNet38ClassificationModel, self).__init__()

        self.mu = 1
        self.gamma = 0.01
        self.enable_PDA = False
        self.num_classes = num_classes

        self.conv1a = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False
        )
        self.conv2a = nn.Conv2d(
            in_channels=3,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False
        )
        self.conv3a = nn.Conv2d(
            in_channels=3,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False
        )
        self.conv4a = nn.Conv2d(
            in_channels=3,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False
        )

        #  -------- B1 -----------#
        # Input size = (224, 224),
        # So no B1 layer

        # -------- B2 -----------#
        self.res2a = ResBlock(
            in_channels=64,
            mid_channels=128,
            out_channels=128,
            first_stride=2,
            first_dilation_padding=1,
            dilation_padding=1
        )
        self.res2b1 = ResBlock(
            in_channels=128,
            mid_channels=128,
            out_channels=128,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1
        )
        self.res2b2 = ResBlock(
            in_channels=128,
            mid_channels=128,
            out_channels=128,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1
        )

        # -------- B3 -----------#
        self.res3a = ResBlock(
            in_channels=128,
            mid_channels=256,
            out_channels=256,
            first_stride=2,
            first_dilation_padding=1,
            dilation_padding=1
        )
        self.res3b1 = ResBlock(
            in_channels=256,
            mid_channels=256,
            out_channels=256,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1
        )
        self.res3b2 = ResBlock(
            in_channels=256,
            mid_channels=256,
            out_channels=256,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1
        )

        # -------- B4 -----------#
        self.res4a = ResBlock(
            in_channels=256,
            mid_channels=512,
            out_channels=512,
            first_stride=2,
            first_dilation_padding=1,
            dilation_padding=1
        )
        self.res4b1 = ResBlock(
            in_channels=512,
            mid_channels=512,
            out_channels=512,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1
        )
        self.res4b2 = ResBlock(
            in_channels=512,
            mid_channels=512,
            out_channels=512,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1
        )
        self.res4b3 = ResBlock(
            in_channels=512,
            mid_channels=512,
            out_channels=512,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1
        )
        self.res4b4 = ResBlock(
            in_channels=512,
            mid_channels=512,
            out_channels=512,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1
        )
        self.res4b5 = ResBlock(
            in_channels=512,
            mid_channels=512,
            out_channels=512,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1
        )

        # -------- B5 -----------#
        self.res5a = ResBlock(
            in_channels=512,
            mid_channels=512,
            out_channels=1024,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=2
        )
        self.res5b1 = ResBlock(
            in_channels=1024,
            mid_channels=512,
            out_channels=1024,
            first_stride=1,
            first_dilation_padding=2,
            dilation_padding=2
        )
        self.res5b2 = ResBlock(
            in_channels=1024,
            mid_channels=512,
            out_channels=1024,
            first_stride=1,
            first_dilation_padding=2,
            dilation_padding=2
        )

        # -------- B6 -----------#
        self.res6a = ResBlockBottleneck(
            in_channels=1024,
            out_channels=2048,
            dilation_padding=4,
            dropout=0.3
        )

        # -------- B7 -----------#
        self.res7a = ResBlockBottleneck(
            in_channels=2048,
            out_channels=4096,
            dilation_padding=4,
            dropout=0.5
        )

        self.bn7 = nn.BatchNorm2d(num_features=4096)
        self.relu7 = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.5)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc8 = nn.Conv2d(in_channels=4096, out_channels=self.num_classes, kernel_size=1, bias=False)

        nn.init.xavier_uniform_(tensor=self.fc8.weight)

        self.not_training = [self.conv1a, self.res2a, self.res2b1, self.res2b2]
        self.from_scratch_layers = [self.fc8]

    def extract_features(self, imgs, enable_PDA=False, cams=None):  # (b, 3, 224, 224)
        b = imgs.size(0)

        patch112 = create_patches(x=imgs, kernel_size=112, stride=56)  # (*9, 3, 112, 112)
        patch56 = create_patches(x=imgs, kernel_size=56, stride=28)  # (*49, 3, 112, 112)
        patch28 = create_patches(x=imgs, kernel_size=28, stride=14)  # (*225, 3, 28, 28)
        print(f'imgs: {imgs.size()}')
        print(f'patch112: {patch112.size()}')
        print(f'patch56: {patch56.size()}')
        print(f'patch28: {patch28.size()}')

        x1 = self.conv1a(imgs)  # (b, 64, 224, 224)
        x2 = self.conv2a(patch112)  # (b, 128, 112, 112)
        x3 = self.conv3a(patch56)  # (b, 256, 56, 56)
        x4 = self.conv4a(patch28)  # (b, 512, 28, 28)
        print(f'x1: {x1.size()}')
        print(f'x2: {x2.size()}')
        print(f'x3: {x3.size()}')
        print(f'x4: {x4.size()}')

        x, x1a = self.res2a(x=x1)  # (b, 128, 112, 112), (b, 64, 224, 224)
        print(f'x: {x.size()}, x1a: {x1a.size()}')
        x = torch.cat(tensors=(x, x2), dim=0)
        print(f'x: {x.size()}')

        x, x2a = self.res2b1(x=x)  # (b, 128, 112, 112), (b, 128, 112, 112)
        print(f'x: {x.size()}, x2a: {x2a.size()}')
        x, x2b1 = self.res2b2(x=x)  # (b, 128, 112, 112), (b, 128, 112, 112)
        print(f'x: {x.size()}, x2b1: {x2b1.size()}')
        x, out2 = x[:b, :, :, :], x[b:, :, :, :]
        print(f'x: {x.size()}, out2: {out2.size()}')

        x, x2b2 = self.res3a(x=x, patches=x3, name='res3a')  # (b, 256, 56, 56), (b, 128, 112, 112)
        print(f'x: {x.size()}, x2b2: {x2b2.size()}')
        x, x3a = self.res3b1(x=x, name='res3b1')  # (b, 256, 56, 56), (b, 256, 56, 56)
        print(f'x: {x.size()}, x3a: {x3a.size()}')
        x, x3b1 = self.res3b2(x=x, name='res3b2')  # (b, 256, 56, 56), (b, 256, 56, 56)
        print(f'x: {x.size()}, x3b1: {x3b1.size()}')
        x, out3 = x[:b, :, :, :], x[b:, :, :, :]

        x, x3b2 = self.res4a(x=x, patches=x4, name='res4a')  # (b, 512, 28, 28), (b, 256, 56, 56)
        print(f'x: {x.size()}, x3b2: {x3b2.size()}')
        x, x4a = self.res4b1(x=x, name='res4b1')  # (b, 512, 28, 28), (b, 512, 28, 28)
        print(f'x: {x.size()}, x4a: {x4a.size()}')
        x, x4b1 = self.res4b2(x=x, name='res4b2')  # (b, 512, 28, 28), (b, 512, 28, 28)
        print(f'x: {x.size()}, x4b1: {x4b1.size()}')
        x, x4b2 = self.res4b3(x=x, name='res4b3')  # (b, 512, 28, 28), (b, 512, 28, 28)
        print(f'x: {x.size()}, x4b2: {x4b2.size()}')
        x, x4b3 = self.res4b4(x=x, name='res4b4')  # (b, 512, 28, 28), (b, 512, 28, 28)
        print(f'x: {x.size()}, x4b3: {x4b3.size()}')
        x, x4b4 = self.res4b5(x=x, name='res4b5')  # (b, 512, 28, 28), (b, 512, 28, 28)
        print(f'x: {x.size()}, x4b4: {x4b4.size()}')

        x, x4b5 = self.res5a(x=x, name='res5a')  # (b, 1024, 28, 28), (b, 512, 28, 28)
        print(f'x: {x.size()}, x4b5: {x4b5.size()}')
        x, x5a = self.res5b1(x=x, name='res5b1')  # (b, 1024, 28, 28), (b, 1024, 28, 28)
        print(f'x: {x.size()}, x5a: {x5a.size()}')
        x, x5b1 = self.res5b2(x=x, name='res5b2')  # (b, 1024, 28, 28), (b, 1024, 28, 28)
        print(f'x: {x.size()}, x5b1: {x5b1.size()}')

        x, x5b2 = self.res6a(x=x, name='res6a')  # (b, 2048, 28, 28), (b, 1024, 28, 28)
        print(f'x: {x.size()}, x5b2: {x5b2.size()}')

        x, x6a = self.res7a(x=x, name='res7a')  # (b, 4096, 28, 28), (b, 2048, 28, 28)
        print(f'x: {x.size()}, x6a: {x6a.size()}')
        x, out4 = x[:b, :, :, :], x[b:, :, :, :]

        x = self.bn7(x)  # (b, 4096, 28, 28)
        print(f'bn7: {x.size()}')
        x = self.relu7(x)  # (b, 4096, 28, 28)
        print(f'relu7: {x.size()}')

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

    # def make_multiscale_cam(self, x):  # (b, 3, 224, 224)
    #     with torch.no_grad():
    #         x7a = self.extract_features(imgs=x)['x7a']  # (b, 4096, 28, 28)
    #         cam28 = torch.conv2d(input=x7a, weight=self.fc8.weight)  # (b, 4, 28, 28)
    #         cam28 = torch.relu(input=cam28)
    #         cam28 = torchutils.standard_scale(x=cam28, dims=-3)
    #
    #         cam56 = F.interpolate(input=cam28, size=(56, 56), mode='bilinear', align_corners=False)
    #         cam56 = torchutils.standard_scale(x=cam56, dims=-3)
    #
    #         cam112 = F.interpolate(input=cam28, size=(112, 112), mode='bilinear', align_corners=False)
    #         cam112 = torchutils.standard_scale(x=cam112, dims=-3)
    #
    #         cam224 = F.interpolate(input=cam28, size=(224, 224), mode='bilinear', align_corners=False)
    #         cam224 = torchutils.standard_scale(x=cam224, dims=-3)
    #     return {'cam28': cam28, 'cam56': cam56, 'cam112': cam112, 'cam224': cam224}  # {'camS': (b, 4, S, S), ...}

    def forward(self, x):  # (b, 3, 224, 224)
        # cams = self.make_multiscale_cam(x=x) if self.enable_PDA else None

        x = self.extract_features(imgs=x, enable_PDA=False, cams=None)['x7a']  # (b, 4096, 56, 56)

        x = self.dropout(x)  # (b, c, h, w)
        x = self.avgpool(x)  # (b, c, 1, 1)
        x = self.fc8(x)  # (b, num_classes, 1, 1)

        return x  # (b, num_classes, 1, 1)

    # def forward_cam(self, x):  # (b, 3, 224, 224)
    #     x = self.extract_features(imgs=x, enable_PDA=False, cams=None)['x7a']  # (b, 4096, 56, 56)
    #
    #     cam = torch.conv2d(input=x, weight=self.fc8.weight)  # (b, 4, 56, 56)
    #     cam = torch.relu(input=cam)  # (b, 4, 56, 56)
    #
    #     z = self.avgpool(x)  # (b, 4096, 1, 1)
    #     z = self.fc8(z)  # (b, 4, 1, 1)
    #     z = torch.sigmoid(input=z)  # (b, 4, 1, 1)
    #
    #     return cam, z  # (b, 4, 56, 56), (b, 4, 1, 1)

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
