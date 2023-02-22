import torch
from torch import nn


class ResBlockBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_padding=1, dropout=0.0):
        super(ResBlockBottleneck, self).__init__()

        self.same_shape = (in_channels == out_channels)

        self.bn_branch2a = nn.BatchNorm2d(num_features=in_channels)
        self.relu_branch2a = nn.ReLU()
        self.conv_branch2a = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=1, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(num_features=out_channels // 4)
        self.relu_branch2b1 = nn.ReLU()
        self.conv_branch2b1 = nn.Conv2d(in_channels=out_channels // 4, out_channels=out_channels // 2, kernel_size=3, padding=dilation_padding, dilation=dilation_padding, bias=False)

        self.bn_branch2b2 = nn.BatchNorm2d(num_features=out_channels // 2)
        self.relu_branch2b2 = nn.ReLU()
        self.dropout_branch2b2 = nn.Dropout2d(p=dropout)
        self.conv_branch2b2 = nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=1, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)

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


class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, first_stride, first_dilation_padding, dilation_padding):
        super(ResBlock, self).__init__()

        self.same_shape = (in_channels == out_channels)

        self.bn_branch2a = nn.BatchNorm2d(num_features=in_channels)
        self.relu_branch2a = nn.ReLU()
        self.conv_branch2a = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=first_stride, padding=first_dilation_padding, dilation=first_dilation_padding, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(num_features=mid_channels)
        self.relu_branch2b1 = nn.ReLU()
        self.conv_branch2b1 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=dilation_padding, dilation=dilation_padding, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=first_stride, bias=False)

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

# def create_patches(x, kernel_size, stride):
#     x = x.unfold(1, 3, 3).unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
#     x = x.contiguous().view(x.size(0), -1, 3, kernel_size, kernel_size)
#     x = x.view(-1, 3, kernel_size, kernel_size)
#     return x


# def intermediate_forward_PDA(x, cam, mu, gamma):
#     # cam = torch.mean(input=x, dim=1, keepdim=True) * (cam > 0)
#
#     beta = torch.amax(input=cam, dim=(-2, -1), keepdim=True) * mu  # (b, num_classes, 1, 1)
#     beta = beta.expand(size=cam.size())  # (b, num_classes, h, w)
#
#     alpha = torch.amax(input=cam, dim=(-2, -1), keepdim=True) * gamma
#     alpha = alpha.expand(size=cam.size())
#
#     cam = torch.less(input=cam, other=beta) * torch.greater(input=cam, other=alpha) * cam  # (b, num_classes, h, w)
#
#     cam = torch.mean(input=cam, dim=1, keepdim=True)  # (b, 1, h, w)
#     x = x * cam  # (b, c, h, w)
#     return x


# def final_forward_PDA(x, w, mu):
#     cam = torch.conv2d(input=x, weight=w)
#     cam = torch.relu(input=cam)
#
#     beta = torch.amax(input=cam, dim=(-2, -1), keepdim=True) * mu  # (b, num_classes, 1, 1)
#     beta = beta.expand(size=cam.size())  # (b, num_classes, h, w)
#
#     cam = torch.less(input=cam, other=beta) * cam  # (b, num_classes, h, w)
#
#     cam = torch.mean(input=cam, dim=1, keepdim=True)  # (b, 1, h, w)
#     x = x * cam  # (b, c, h, w)
#     return x





# def forward_PDA(x, mu):
#     cam = torch.mean(input=x, dim=1, keepdim=True)  # (b, 1, h, w)
#     beta = torch.amax(input=cam, dim=(-2, -1), keepdim=True) * mu  # (b, 1, 1, 1)
#     beta = beta.expand(size=cam.size())  # (b, 1, h, w)
#     x = torch.less(input=cam, other=beta) * x  # (b, c, h, w)
#     return x  # (b, c, h, w)


# class ResBlock(nn.Module):
#     def __init__(self, in_channels, mid_channels, out_channels, stride=1, first_dilation=None, dilation=1):
#         super(ResBlock, self).__init__()
#
#         self.same_shape = (in_channels == out_channels and stride == 1)
#
#         if first_dilation is None: first_dilation = dilation
#
#         self.bn_branch2a = nn.BatchNorm2d(num_features=in_channels)
#         self.relu_branch2a = nn.ReLU()
#         self.conv_branch2a = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=mid_channels,
#             kernel_size=3,
#             stride=stride,
#             padding=first_dilation,
#             dilation=first_dilation,
#             bias=False
#         )
#
#         self.bn_branch2b1 = nn.BatchNorm2d(num_features=mid_channels)
#         self.relu_branch2b1 = nn.ReLU()
#         self.conv_branch2b1 = nn.Conv2d(
#             in_channels=mid_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             padding=dilation,
#             dilation=dilation,
#             bias=False
#         )
#
#         if not self.same_shape:
#             self.conv_branch1 = nn.Conv2d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=1,
#                 stride=stride,
#                 bias=False
#             )
#
#     def forward_block(self, branch2):
#         branch2 = self.conv_branch2a(branch2)
#         branch2 = self.bn_branch2b1(branch2)
#         branch2 = self.relu_branch2b1(branch2)
#         branch2 = self.conv_branch2b1(branch2)
#         return branch2
#
#     def forward(self, x, mu=None):
#
#         branch2 = self.bn_branch2a(x)
#         branch2 = self.relu_branch2a(branch2)
#
#         x_bn_relu = branch2
#
#         if not self.same_shape:
#             branch1 = self.conv_branch1(branch2)
#         else:
#             branch1 = x
#
#         branch2_out = self.forward_block(branch2=branch2)
#
#         if mu is not None:
#             branch2_out = branch2_out + self.forward_block(branch2=forward_PDA(x=branch2, mu=mu))
#
#         x = branch1 + branch2_out
#
#         return x, x_bn_relu
#
#
# class ResBlockBot(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.0):
#         super(ResBlockBot, self).__init__()
#
#         self.same_shape = (in_channels == out_channels and stride == 1)
#
#         self.bn_branch2a = nn.BatchNorm2d(num_features=in_channels)
#         self.relu_branch2a = nn.ReLU()
#         self.conv_branch2a = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=out_channels // 4,
#             kernel_size=1,
#             stride=stride,
#             bias=False
#         )
#
#         self.bn_branch2b1 = nn.BatchNorm2d(num_features=out_channels // 4)
#         self.relu_branch2b1 = nn.ReLU()
#         self.dropout_2b1 = nn.Dropout2d(p=dropout)
#         self.conv_branch2b1 = nn.Conv2d(
#             in_channels=out_channels // 4,
#             out_channels=out_channels // 2,
#             kernel_size=3,
#             padding=dilation,
#             dilation=dilation,
#             bias=False
#         )
#
#         self.bn_branch2b2 = nn.BatchNorm2d(num_features=out_channels // 2)
#         self.relu_branch2b2 = nn.ReLU()
#         self.dropout_2b2 = nn.Dropout2d(p=dropout)
#         self.conv_branch2b2 = nn.Conv2d(
#             in_channels=out_channels // 2,
#             out_channels=out_channels,
#             kernel_size=1,
#             bias=False
#         )
#
#         if not self.same_shape:
#             self.conv_branch1 = nn.Conv2d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=1,
#                 stride=stride,
#                 bias=False
#             )
#
#     def forward_block(self, branch2):
#
#         branch2 = self.conv_branch2a(branch2)  # (b, 1024, 28, 28)
#
#         branch2 = self.bn_branch2b1(branch2)  # (b, 1024, 28, 28)
#         branch2 = self.relu_branch2b1(branch2)  # (b, 1024, 28, 28)
#         branch2 = self.dropout_2b1(branch2)  # (b, 1024, 28, 28)
#         branch2 = self.conv_branch2b1(branch2)  # (b, 2048, 28, 28)
#
#         branch2 = self.bn_branch2b2(branch2)  # (b, 2048, 28, 28)
#         branch2 = self.relu_branch2b2(branch2)  # (b, 2048, 28, 28)
#         branch2 = self.dropout_2b2(branch2)  # (b, 2048, 28, 28)
#         branch2 = self.conv_branch2b2(branch2)  # (b, 4096, 28, 28)
#
#         return branch2
#
#     def forward(self, x, mu=None):  # (b, 2048, 28, 28)
#
#         branch2 = self.bn_branch2a(x)  # (b, 2048, 28, 28)
#         branch2 = self.relu_branch2a(branch2)  # (b, 2048, 28, 28)
#
#         x_bn_relu = branch2
#
#         branch1 = self.conv_branch1(branch2)  # (b, 4096, 28, 28)
#
#         branch2_out = self.forward_block(branch2=branch2)
#
#         if mu is not None:
#             branch2_out = branch2_out + self.forward_block(branch2=forward_PDA(x=branch2, mu=mu))
#
#         x = branch1 + branch2_out  # (b, 4096, 28, 28)
#
#         return x, x_bn_relu  # (b, 4096, 28, 28)


# def forward_PDA(x, w, mu):  # (b, c, h, w), (c, num_classes, 1, 1), int
#
#     cams = torch.conv2d(input=x, weight=w)  # (b, num_classes, h, w)
#     cams = torch.relu(input=cams)  # (b, num_classes, h, w)
#     beta = torch.amax(input=cams, dim=(-2, -1), keepdim=True) * mu  # (b, num_classes, 1, 1)
#     beta = beta.expand(size=cams.size())  # (b, num_classes, h, w)
#     cams = torch.less(input=cams, other=beta) * cams  # (b, num_classes, h, w)
#     cams = torch.mean(input=cams, dim=1, keepdim=True)  # (b, 1, h, w)
#     cams = x * cams  # (b, c, h, w)
#
#     return cams  # (b, c, h, w)


# class MultiHeadAttention(nn.Module):
#     def __init__(self, q_dim, kv_dim, num_heads, attn_drop=0.0, proj_drop=0.0):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = q_dim // self.num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.Q = nn.Linear(in_features=q_dim, out_features=q_dim, bias=False)
#         self.K = nn.Linear(in_features=kv_dim, out_features=q_dim, bias=False)
#         self.V = nn.Linear(in_features=kv_dim, out_features=q_dim, bias=False)
#         self.attn_drop = nn.Dropout(p=attn_drop)
#         self.proj = nn.Conv2d(in_channels=q_dim, out_channels=q_dim, kernel_size=1, bias=False)
#         self.proj_drop = nn.Dropout2d(p=proj_drop)
#
#     def forward(self, q, k, v):  # (b, c_q, h_q, w_q), (b, c_k, h_k, w_k), (b, c_v, h_v, w_v)
#
#         b, c_q, h_q, w_q = q.size()
#         n_q = h_q * w_q
#
#         assert k.size(2) == v.size(2)
#         assert k.size(3) == v.size(3)
#
#         _, c_k, h_kv, w_kv = k.size()
#         n_kv = h_kv * w_kv
#         c_v = v.size(1)
#
#         q = q.view(b, c_q, n_q)  # (b, c_q, h_q, w_q) -> (b, c_q, n_q)
#         k = k.view(b, c_k, n_kv)  # (b, c_k, h_kv, w_kv) -> (b, c_k, n_kv)
#         v = v.view(b, c_v, n_kv)  # (b, c_v, h_kv, w_kv) -> (b, c_v, n_kv)
#
#         q = q.transpose(-2, -1)  # (b, c_q, n_q) -> (b, n_q, c_q)
#         k = k.transpose(-2, -1)  # (b, c_k, n_kv) -> (b, n_kv, c_k)
#         v = v.transpose(-2, -1)  # (b, c_v, n_kv) -> (b, n_kv, c_v)
#
#         q = self.Q(q)  # (b, n_q, c_q) -> (b, n_q, c_q)
#         k = self.K(k)  # (b, n_kv, c_k) -> (b, n_kv, c_q)
#         v = self.V(v)  # (b, n_kv, c_v) -> (b, n_kv, c_q)
#
#         q = q.reshape(b, n_q, self.num_heads, self.head_dim)  # (b, n_q, c_q) -> (b, n_q, num_heads, head_dim)
#         k = k.reshape(b, n_kv, self.num_heads, self.head_dim)  # (b, n_kv, c_q) -> (b, n_kv, num_heads, head_dim)
#         v = v.reshape(b, n_kv, self.num_heads, self.head_dim)  # (b, n_kv, c_q) -> (b, n_kv, num_heads, head_dim)
#
#         q = q.transpose(1, 2)  # (b, n_q, num_heads, head_dim) -> (b, num_heads, n_q, head_dim)
#         k = k.transpose(1, 2)  # (b, n_kv, num_heads, head_dim) -> (b, num_heads, n_kv, head_dim)
#         v = v.transpose(1, 2)  # (b, n_kv, num_heads, head_dim) -> (b, num_heads, n_kv, head_dim)
#
#         # (b, num_heads, n_q, head_dim) @ (b, num_heads, head_dim, n_kv) -> (b, num_heads, n_q, n_kv)
#         attn = q @ k.transpose(-2, -1)
#         attn = attn * self.scale  # (b, num_heads, n_q, n_kv)
#         attn = attn.softmax(dim=-1)  # (b, num_heads, n_q, n_kv)
#         attn = self.attn_drop(attn)  # (b, num_heads, n_q, n_kv)
#
#         # (b, num_heads, n_q, n_kv) @ (b, num_heads, n_kv, head_dim) -> (b, num_heads, n_q, head_dim)
#         x = attn @ v
#         x = x.transpose(1, 2)  # (b, num_heads, n_q, head_dim) -> (b, num_heads, head_dim, n_q)
#         x = x.reshape(b, c_q, n_q)  # (b, num_heads, head_dim, n_q) -> (b, c_q, n_q)
#         x = x.reshape(b, c_q, h_q, w_q)  # (b, c_q, n_q) -> (b, c_q, h_q, w_q)
#         x = self.proj(x)  # (b, c_q, h_q, w_q) -> (b, c_q, h_q, w_q)
#         x = self.proj_drop(x)  # (b, c_q, h_q, w_q)
#
#         return x  # (b, c_q, h_q, w_q)
