import torch
from torch import nn
from src.models.utils import ResBlock, ResBlockBottleneck


class ResNet38ClassificationModel(nn.Module):
    def __init__(self):
        super(ResNet38ClassificationModel, self).__init__()

        self.conv1a = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False)

        #  -------- B1 -----------#
        # Input size = (224, 224),
        # So no B1 layer

        # -------- B2 -----------#
        self.res2a = ResBlock(in_channels=64, mid_channels=128, out_channels=128, first_stride=2, first_dilation_padding=1, dilation_padding=1)
        self.res2b1 = ResBlock(in_channels=128, mid_channels=128, out_channels=128, first_stride=1, first_dilation_padding=1, dilation_padding=1)
        self.res2b2 = ResBlock(in_channels=128, mid_channels=128, out_channels=128, first_stride=1, first_dilation_padding=1, dilation_padding=1)

        # -------- B3 -----------#
        self.res3a = ResBlock(in_channels=128, mid_channels=256, out_channels=256, first_stride=2, first_dilation_padding=1, dilation_padding=1)
        self.res3b1 = ResBlock(in_channels=256, mid_channels=256, out_channels=256, first_stride=1, first_dilation_padding=1, dilation_padding=1)
        self.res3b2 = ResBlock(in_channels=256, mid_channels=256, out_channels=256, first_stride=1, first_dilation_padding=1, dilation_padding=1)

        # -------- B4 -----------#
        self.res4a = ResBlock(in_channels=256, mid_channels=512, out_channels=512, first_stride=2, first_dilation_padding=1, dilation_padding=1)
        self.res4b1 = ResBlock(in_channels=512, mid_channels=512, out_channels=512, first_stride=1, first_dilation_padding=1, dilation_padding=1)
        self.res4b2 = ResBlock(in_channels=512, mid_channels=512, out_channels=512, first_stride=1, first_dilation_padding=1, dilation_padding=1)
        self.res4b3 = ResBlock(in_channels=512, mid_channels=512, out_channels=512, first_stride=1, first_dilation_padding=1, dilation_padding=1)
        self.res4b4 = ResBlock(in_channels=512, mid_channels=512, out_channels=512, first_stride=1, first_dilation_padding=1, dilation_padding=1)
        self.res4b5 = ResBlock(in_channels=512, mid_channels=512, out_channels=512, first_stride=1, first_dilation_padding=1, dilation_padding=1)

        # -------- B5 -----------#
        self.res5a = ResBlock(in_channels=512, mid_channels=512, out_channels=1024, first_stride=1, first_dilation_padding=1, dilation_padding=2)
        self.res5b1 = ResBlock(in_channels=1024, mid_channels=512, out_channels=1024, first_stride=1, first_dilation_padding=2, dilation_padding=2)
        self.res5b2 = ResBlock(in_channels=1024, mid_channels=512, out_channels=1024, first_stride=1, first_dilation_padding=2, dilation_padding=2)

        # -------- B6 -----------#
        self.res6a = ResBlockBottleneck(in_channels=1024, out_channels=2048, dilation_padding=4, dropout=0.3)

        # -------- B7 -----------#
        self.res7a = ResBlockBottleneck(in_channels=2048, out_channels=4096, dilation_padding=4, dropout=0.5)
        # -----------------------#

        self.bn7 = nn.BatchNorm2d(num_features=4096)
        self.relu7 = nn.ReLU()

    def extract_features(self, imgs):

        x1 = self.conv1a(imgs)  # (b, 64, 224, 224)

        # -------- B2 -----------#
        x, x1a = self.res2a(x=x1)  # (b, 128, 112, 112), (b, 64, 224, 224)
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

    def forward(self, x):
        x = self.extract_features(imgs=x)['x7a']  # (b, 4096, 28, 28)
        x = self.dropout(x)  # (b, 4096, 28, 28)
        x = self.avgpool(x)  # (b, 4096, 1, 1)
        x = self.fc8(x)  # (b, 4, 1, 1)

        return x  # (b, 4, 1, 1)

    def forward_cam(self, x):  # (b, 3, 224, 224)
        x = self.extract_features(imgs=x)['x7a']  # (b, 4096, 28, 28)

        cam = torch.conv2d(input=x, weight=self.fc8.weight)  # (b, 4, 28, 28)
        cam = torch.relu(input=cam)  # (b, 4, 28, 28)

        z = self.avgpool(x)  # (b, 4096, 1, 1)
        z = self.fc8(z)  # (b, 4, 1, 1)
        z = torch.sigmoid(input=z)  # (b, 4, 1, 1)

        return cam, z  # (b, 4, 28, 28), (b, 4, 1, 1)


if __name__ == '__main__':
    model = ResNet38ClassificationModel()
    inputs = torch.ones((4, 3, 224, 224))

    z = model(inputs)