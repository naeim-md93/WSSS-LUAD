import torch
from torch import nn
from torch.utils import model_zoo
import src.models.utils.custom_layers as CL
from src.utils import imgutils


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = CL.conv1x1(in_planes=inplanes, out_planes=planes)
        self.bn1 = CL.BatchNorm2d(num_features=planes)
        self.conv2 = CL.conv3x3(in_planes=planes, out_planes=planes, stride=stride)
        self.bn2 = CL.BatchNorm2d(num_features=planes)
        self.conv3 = CL.conv1x1(in_planes=planes, out_planes=planes * self.expansion)
        self.bn3 = CL.BatchNorm2d(num_features=planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        self.relu1 = CL.ReLU()
        self.relu2 = CL.ReLU()
        self.relu3 = CL.ReLU()

        self.add = CL.Add()

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            x = self.downsample(x)
        else:
            x = x

        out = self.add(out, x)
        out = self.relu3(out)

        return out

    def relprop(self, R, alpha):
        out = self.relu3.relprop(R, alpha)

        out, x = self.add.relprop(out, alpha)

        if self.downsample is not None:
            x = self.downsample.relprop(x, alpha)

        out = self.bn3.relprop(out, alpha)
        out = self.conv3.relprop(out, alpha)

        out = self.relu2.relprop(out, alpha)
        out = self.bn2.relprop(out, alpha)
        out = self.conv2.relprop(out, alpha)

        out = self.relu1.relprop(out, alpha)
        out = self.bn1.relprop(out, alpha)
        x1 = self.conv1.relprop(out, alpha)

        return x1 + x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, device):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.mu = 1
        self.enable_PDA = True
        self.device = device

        self.inplanes = 64
        self.conv1 = CL.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = CL.BatchNorm2d(num_features=64)
        self.relu = CL.ReLU()
        self.maxpool = CL.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.dropout = CL.Dropout2d(p=0.5)
        self.avgpool = CL.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = CL.Linear(in_features=512 * block.expansion, out_features=1000)

        nn.init.xavier_uniform_(tensor=self.fc.weight)
        self.not_training = [self.conv1]
        self.from_scratch_layers = [self.fc]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = CL.Sequential(
                CL.conv1x1(in_planes=self.inplanes, out_planes=planes * block.expansion, stride=stride),
                CL.BatchNorm2d(num_features=planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return CL.Sequential(*layers)

    def extract_features(self, x):  # (b, 3, 224, 224)
        x = self.conv1(x)  # (b, 64, 112, 112)
        x = self.bn1(x)  # (b, 64, 112, 112)
        x = self.relu(x)  # (b, 64, 112, 112)
        x = self.maxpool(x)  # (b, 64, 56, 56)

        x1 = self.layer1(x)  # (b, 256, 56, 56)
        x2 = self.layer2(x1)  # (b, 512, 28, 28)
        x3 = self.layer3(x2)  # (b, 1024, 14, 14)
        x4 = self.layer4(x3)  # (b, 2048, 7, 7)

        return {'layer1': x1, 'layer2': x2, 'layer3': x3, 'layer4': x4}

    def classifier(self, x):  # (b, 2048, 7, 7)
        x = self.dropout(x)  # (b, 2048, 7, 7)
        x = self.avgpool(x)  # (b, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # (b, 2048)
        x = self.fc(x)  # (b, 4)
        return x  # (b, 4)

    def clrp(self, z, target_class):
        R = torch.ones(size=z.size(), device=self.device)  # (b, 4)
        R = R / -self.num_classes  # (b, 4)
        R[:, target_class] = 1
        # R = torch.zeros_like(input=z)
        # R[:, target_class] = z[:, target_class]
        return R  # (b, 4)

    def forward_relevance_cam(self, f):  # (b, 2048, 7, 7)
        cams = {
            'layer4': [],
            'layer3': [],
            'layer2': [],
            'layer1': [],
        }
        x1 = f['layer1']
        x2 = f['layer2']
        x3 = f['layer3']
        x4 = f['layer4']
        z = self.classifier(x4)  # (b, 4)

        for i in range(self.num_classes):
            R = self.clrp(z=z, target_class=i)
            R = self.fc.relprop(R=R, alpha=1)
            R = R.reshape(R.size(0), R.size(1), 1, 1)
            R4 = self.avgpool.relprop(R=R, alpha=1)
            R3 = self.layer4.relprop(R4, alpha=1)
            R2 = self.layer3.relprop(R3, alpha=1)
            R1 = self.layer2.relprop(R2, alpha=1)

            cams['layer4'].append(imgutils.grad_cam(a=x4, g=R4))
            cams['layer3'].append(imgutils.grad_cam(a=x3, g=R3))
            cams['layer2'].append(imgutils.grad_cam(a=x2, g=R2))
            cams['layer1'].append(imgutils.grad_cam(a=x1, g=R1))

        cams = {k: torch.cat(tensors=v, dim=1) for k, v in cams.items()}
        return cams, z

    def forward_PDA(self, f):  # (b, 2048, 7, 7)
        if self.enable_PDA:
            cams, z = self.forward_relevance_cam(f=f)  # (b, 4, 56, 56)
            cam = cams['layer1']

            beta = torch.amax(input=cam, dim=(-2, -1), keepdim=True) * self.mu  # (b, 4, 1, 1)
            beta = beta.expand(size=cam.size())  # (b, 4, 56, 56)
            cam = torch.le(input=cam, other=beta) * cam  # (b, 4, 56, 56)
            cam = torch.mean(input=cam, dim=1, keepdim=True)  # (b, 1, 56, 56)
            x = f['layer1'] * cam  # (b, 256, 56, 56)
            x = self.layer2(x)  # (b, 512, 28, 28)
            x = self.layer3(x)  # (b, 1024, 14, 14)
            x = self.layer4(x)  # (b, 2048, 7, 7)
            f['layer4'] = x

        return f['layer4']  # (b, 2048, 7, 7)

    def forward(self, x):  # (b, 3, 224, 224)
        f = self.extract_features(x=x)  # (b, 2048, 7, 7)
        x4 = self.forward_PDA(f=f)  # (b, 2048, 7, 7)
        z = self.classifier(x=x4)  # (b, 4)
        return z

    def forward_cam(self, x):  # (b, 3, 224, 224)
        f = self.extract_features(x=x)  # (b, 2048, 7, 7)
        cam, z = self.forward_relevance_cam(f=f)  # (b, ?, ?, ?), (b, 4)
        return cam, z  # (b, ?, ?, ?), (b, 4)

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


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def resnet50(pretrained=False,**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(state_dict=model_zoo.load_url(url=model_urls['resnet50']), strict=False)
    model.fc = CL.Linear(in_features=2048, out_features=kwargs['num_classes'])
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(state_dict=model_zoo.load_url(url=model_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(block=Bottleneck, layers=[3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(state_dict=model_zoo.load_url(url=model_urls['resnet152']), strict=False)
    return model