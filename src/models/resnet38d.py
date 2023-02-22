from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, first_stride=1, first_dilation_padding=1, dilation_padding=1):
        super(ResBlock, self).__init__()

        self.same_shape = (in_channels == out_channels)

        self.bn_branch2a = nn.BatchNorm2d(num_features=in_channels)
        self.relu_branch2a = nn.ReLU()
        self.conv_branch2a = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=first_stride,
            padding=first_dilation_padding,
            dilation=first_dilation_padding,
            groups=1,
            bias=False
        )

        self.bn_branch2b1 = nn.BatchNorm2d(num_features=mid_channels)
        self.relu_branch2b1 = nn.ReLU()
        self.conv_branch2b1 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation_padding,
            dilation=dilation_padding,
            groups=1,
            bias=False
        )

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=first_stride,
                padding=0,
                dilation=1,
                groups=1,
                bias=False
            )

    def forward(self, x):

        branch2 = self.bn_branch2a(x)
        branch2 = self.relu_branch2a(branch2)

        x_bn_relu = branch2

        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)
        else:
            branch1 = x

        branch2 = self.conv_branch2a(branch2)

        branch2 = self.bn_branch2b1(branch2)
        branch2 = self.relu_branch2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)

        x = branch2 + branch1

        return x, x_bn_relu


class ResBlockBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_padding=1, dropout=0.0):
        super(ResBlockBottleneck, self).__init__()

        self.same_shape = (in_channels == out_channels)

        self.bn_branch2a = nn.BatchNorm2d(num_features=in_channels)
        self.relu_branch2a = nn.ReLU()
        self.conv_branch2a = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False
        )

        self.bn_branch2b1 = nn.BatchNorm2d(num_features=out_channels // 4)
        self.relu_branch2b1 = nn.ReLU()
        self.conv_branch2b1 = nn.Conv2d(
            in_channels=out_channels // 4,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=1,
            padding=dilation_padding,
            dilation=dilation_padding,
            groups=1,
            bias=False
        )

        self.bn_branch2b2 = nn.BatchNorm2d(num_features=out_channels // 2)
        self.relu_branch2b2 = nn.ReLU()
        self.dropout_branch2b2 = nn.Dropout2d(p=dropout)
        self.conv_branch2b2 = nn.Conv2d(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False
        )

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False
            )

    def forward(self, x):

        branch2 = self.bn_branch2a(x)
        branch2 = self.relu_branch2a(branch2)

        x_bn_relu = branch2

        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)

        else:
            branch1 = x

        branch2 = self.conv_branch2a(branch2)

        branch2 = self.bn_branch2b1(branch2)
        branch2 = self.relu_branch2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)

        branch2 = self.bn_branch2b2(branch2)
        branch2 = self.relu_branch2b2(branch2)
        branch2 = self.conv_branch2b2(branch2)

        x = branch1 + branch2

        return x, x_bn_relu


class ResNet38(nn.Module):
    def __init__(self):
        super(ResNet38, self).__init__()

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
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear1000 = nn.Linear(
            in_features=4096,
            out_features=1000,
            bias=True
        )

    def extract_features(self, x):

        x = self.conv1a(x)  # (b, 64, 224, 224)

        x, x1a = self.res2a(x)  # (b, 128, 112, 112), (b, 64, 224, 224)
        x, x2a = self.res2b1(x)  # (b, 128, 112, 112), (b, 128, 112, 112)
        x, x2b1 = self.res2b2(x)  # (b, 128, 112, 112), (b, 128, 112, 112)

        x, x2b2 = self.res3a(x)  # (b, 256, 56, 56), (b, 128, 112, 112)
        x, x3a = self.res3b1(x)  # (b, 256, 56, 56), (b, 256, 56, 56)
        x, x3b1 = self.res3b2(x)  # (b, 256, 56, 56), (b, 256, 56, 56)

        x, x3b2 = self.res4a(x)  # (b, 512, 28, 28), (b, 256, 56, 56)
        x, x4a = self.res4b1(x)  # (b, 512, 28, 28), (b, 512, 28, 28)
        x, x4b1 = self.res4b2(x)  # (b, 512, 28, 28), (b, 512, 28, 28)
        x, x4b2 = self.res4b3(x)  # (b, 512, 28, 28), (b, 512, 28, 28)
        x, x4b3 = self.res4b4(x)  # (b, 512, 28, 28), (b, 512, 28, 28)
        x, x4b4 = self.res4b5(x)  # (b, 512, 28, 28), (b, 512, 28, 28)

        x, x4b5 = self.res5a(x)  # (b, 1024, 28, 28), (b, 512, 28, 28)
        x, x5a = self.res5b1(x)  # (b, 1024, 28, 28), (b, 1024, 28, 28)
        x, x5b1 = self.res5b2(x)  # (b, 1024, 28, 28), (b, 1024, 28, 28)

        x, x5b2 = self.res6a(x)  # (b, 2048, 28, 28), (b, 1024, 28, 28)

        x, x6a = self.res7a(x)  # (b, 4096, 28, 28), (b, 2048, 28, 28)

        x = self.bn7(x)  # (b, 4096, 28, 28), (b, 4096, 28, 28)
        x = self.relu7(x)  # (b, 4096, 28, 28), (b, 4096, 28, 28)

        # x = self.avgpool(x).view(x.size(0), -1)  # (b, 4096)
        # x = self.linear1000(x)  # (b, 1000)

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
