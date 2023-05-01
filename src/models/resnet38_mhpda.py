import torch
import mxnet
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q_linear = nn.Linear(in_features=q_dim, out_features=q_dim, bias=qkv_bias)
        self.k_linear = nn.Linear(in_features=k_dim, out_features=q_dim, bias=qkv_bias)
        self.v_linear = nn.Linear(in_features=v_dim, out_features=q_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(in_features=q_dim, out_features=q_dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def forward(self, q, k, v):  # (b, 4096, 28, 28), (b, 4, 28, 28), (b, 4, 28, 28)

        # q = (b, 4096, 28, 28)
        (b, c_q, h_q, w_q), n_q = q.size(), q.size(2) * q.size(3)
        # (b, 4096, 28, 28) -> (b, 4096, 784) -> (b, 784, 4096)
        q = q.view(b, c_q, n_q).transpose(dim0=-1, dim1=-2)
        # (b, 784, 4096) -> (b, 784, 4096)
        q = self.q_linear(q)
        # (b, 784, 4096) -> (b, 784, 4, 1024) -> (b, 4, 784, 1024)
        q = q.reshape(b, n_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # k = (b, 4, 28, 28)
        (_, c_k, h_k, w_k), n_k = k.size(), k.size(2) * k.size(3)
        # (b, 4, 28, 28) -> (b, 4, 784) -> (b, 784, 4)
        k = k.view(b, c_k, n_k).transpose(dim0=-1, dim1=-2)
        # (b, 784, 4) -> (b, 784, 4096)
        k = self.k_linear(k)
        # (b, 784, 4096) -> (b, 784, 4, 1024) -> (b, 4, 784, 1024)
        k = k.reshape(b, n_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # v = (b, 4, 28, 28)
        (_, c_v, h_v, w_v), n_v = v.size(), v.size(2) * v.size(3)
        # (b, 4, 28, 28) -> (b, 4, 784) -> (b, 784, 4)
        v = v.view(b, c_v, n_v).transpose(dim0=1, dim1=2)
        # (b, 784, 4) -> (b, 784, 4096)
        v = self.v_linear(v)
        # (b, 784, 4096) -> (b, 784, 4, 1024) -> (b, 4, 784, 1024)
        v = v.reshape(b, n_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # (b, 4, 784, 1024) @ (b, 4, 1024, 784) -> (b, 4, 784, 784)
        attn = (q @ k.transpose(dim0=-2, dim1=-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (b, 4, 784, 784) @ (b, 4, 784, 1024) -> (b, 4, 784, 1024) -> (b, 784, 4, 1024) -> (b, 784, 4096)
        x = (attn @ v).transpose(dim0=1, dim1=2).reshape(b, n_q, c_q)

        x = self.proj(x)  # (b, 784, 4096) -> (b, 784, 4096)
        x = self.proj_drop(x)

        # (b, 784, 4096) -> (b, 4096, 784) -> (b, 4096, 28, 28)
        x = x.transpose(dim0=-1, dim1=-2).reshape(b, c_q, h_q, w_q)

        return x


class MultiHeadProgressiveDropoutAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads):
        super().__init__()
        self.MHA = MultiHeadAttention(q_dim=q_dim, k_dim=kv_dim, v_dim=kv_dim, num_heads=num_heads)

    def forward(self, x, w):
        x = self.MHA(q=x, k=w, v=w)
        return x


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


class ResNet38MHPDA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

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
            dilation_padding=1,
        )
        self.res2b1 = ResBlock(
            in_channels=128,
            mid_channels=128,
            out_channels=128,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1,
        )
        self.res2b2 = ResBlock(
            in_channels=128,
            mid_channels=128,
            out_channels=128,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1,
        )

        # -------- B3 -----------#
        self.res3a = ResBlock(
            in_channels=128,
            mid_channels=256,
            out_channels=256,
            first_stride=2,
            first_dilation_padding=1,
            dilation_padding=1,
        )
        self.res3b1 = ResBlock(
            in_channels=256,
            mid_channels=256,
            out_channels=256,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1,
        )
        self.res3b2 = ResBlock(
            in_channels=256,
            mid_channels=256,
            out_channels=256,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1,
        )

        # -------- B4 -----------#
        self.res4a = ResBlock(
            in_channels=256,
            mid_channels=512,
            out_channels=512,
            first_stride=2,
            first_dilation_padding=1,
            dilation_padding=1,
        )
        self.res4b1 = ResBlock(
            in_channels=512,
            mid_channels=512,
            out_channels=512,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1,
        )
        self.res4b2 = ResBlock(
            in_channels=512,
            mid_channels=512,
            out_channels=512,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1,
        )
        self.res4b3 = ResBlock(
            in_channels=512,
            mid_channels=512,
            out_channels=512,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1,
        )
        self.res4b4 = ResBlock(
            in_channels=512,
            mid_channels=512,
            out_channels=512,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1,
        )
        self.res4b5 = ResBlock(
            in_channels=512,
            mid_channels=512,
            out_channels=512,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=1,
        )

        # -------- B5 -----------#
        self.res5a = ResBlock(
            in_channels=512,
            mid_channels=512,
            out_channels=1024,
            first_stride=1,
            first_dilation_padding=1,
            dilation_padding=2,
        )
        self.res5b1 = ResBlock(
            in_channels=1024,
            mid_channels=512,
            out_channels=1024,
            first_stride=1,
            first_dilation_padding=2,
            dilation_padding=2,
        )
        self.res5b2 = ResBlock(
            in_channels=1024,
            mid_channels=512,
            out_channels=1024,
            first_stride=1,
            first_dilation_padding=2,
            dilation_padding=2,
        )

        # -------- B6 -----------#
        self.res6a = ResBlockBottleneck(
            in_channels=1024,
            out_channels=2048,
            dilation_padding=4,
            dropout=0.3,
        )

        # -------- B7 -----------#
        self.res7a = ResBlockBottleneck(
            in_channels=2048,
            out_channels=4096,
            dilation_padding=4,
            dropout=0.5,
        )
        self.bn7 = nn.BatchNorm2d(num_features=4096)
        self.relu7 = nn.ReLU()
        self.MHPDA7 = MultiHeadProgressiveDropoutAttention(q_dim=4096, kv_dim=4, num_heads=4)
        self.dropout7 = nn.Dropout2d(p=0.5)
        # -----------------------#

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc8 = nn.Conv2d(in_channels=4096, out_channels=num_classes, kernel_size=1, bias=False)

        nn.init.xavier_uniform_(tensor=self.fc8.weight)
        self.not_training = [self.conv1a, self.res2a, self.res2b1, self.res2b2]
        self.from_scratch_layers = [self.MHPDA7, self.fc8]

    def extract_features(self, imgs):

        # -------- B0 -----------#
        x = self.conv1a(imgs)  # (b, 64, 224, 224)
        # -----------------------#

        # -------- B2 -----------#
        x, x1a = self.res2a(x=x)  # (b, 128, 112, 112), (b, 64, 224, 224)
        x, x2a = self.res2b1(x=x)  # (b, 128, 112, 112), (b, 128, 112, 112)
        x, x2b1 = self.res2b2(x=x)  # (b, 128, 112, 112), (b, 128, 112, 112)
        # -----------------------#

        # -------- B3 -----------#
        x, x2b2 = self.res3a(x=x)  # (b, 256, 56, 56), (b, 128, 112, 112)
        x, x3a = self.res3b1(x=x)  # (b, 256, 56, 56), (b, 256, 56, 56)
        x, x3b1 = self.res3b2(x=x)  # (b, 256, 56, 56), (b, 256, 56, 56)
        # -----------------------#

        # -------- B4 -----------#
        x, x3b2 = self.res4a(x=x)  # (b, 512, 28, 28), (b, 256, 56, 56)
        x, x4a = self.res4b1(x=x)  # (b, 512, 28, 28), (b, 512, 28, 28)
        x, x4b1 = self.res4b2(x=x)  # (b, 512, 28, 28), (b, 512, 28, 28)
        x, x4b2 = self.res4b3(x=x)  # (b, 512, 28, 28), (b, 512, 28, 28)
        x, x4b3 = self.res4b4(x=x)  # (b, 512, 28, 28), (b, 512, 28, 28)
        x, x4b4 = self.res4b5(x=x)  # (b, 512, 28, 28), (b, 512, 28, 28)
        # -----------------------#

        # -------- B5 -----------#
        x, x4b5 = self.res5a(x=x)  # (b, 1024, 28, 28), (b, 512, 28, 28)
        x, x5a = self.res5b1(x=x)  # (b, 1024, 28, 28), (b, 1024, 28, 28)
        x, x5b1 = self.res5b2(x=x)  # (b, 1024, 28, 28), (b, 1024, 28, 28)
        # -----------------------#

        # -------- B6 -----------#
        x, x5b2 = self.res6a(x=x)  # (b, 2048, 28, 28), (b, 1024, 28, 28)
        # -----------------------#

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

    def create_weighted_cam(self, x):  # (b, 4096, 28, 28)
        with torch.no_grad():
            x = F.conv2d(input=x, weight=self.fc8.weight)  # (b, 4, 28, 28)
        return x

    def forward(self, imgs):  # (b, 3, 224, 224)
        x = self.extract_features(imgs=imgs)['x7a']  # (b, 4096, 28, 28), (b, 4, 28, 28)
        weighted_cam = self.create_weighted_cam(x=x)  # (b, 4, 28, 28)
        x = self.MHPDA7(x=x, w=weighted_cam) # (b, 4096, 28, 28)

        # -------- B7 -----------#
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
        super().train(mode)

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
        print(f'{"#" * 10} Loading Model Weights {"#" * 10}')
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
