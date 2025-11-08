from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModelArgs, NNModel


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class NNModel_ResNet9(NNModel):
    """
    ResNet9 model for image classification
    Simplified ResNet architecture with 9 layers
    """

    def __init__(self):
        super().__init__()

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        in_channels = int(getattr(args, "in_channels", 3))
        num_classes = int(getattr(args, "num_classes", 10))

        # ResNet9 architecture
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Layer 1
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 1, stride=1)
        # Layer 2
        self.layer2 = self._make_layer(BasicBlock, 64, 128, 1, stride=2)
        # Layer 3
        self.layer3 = self._make_layer(BasicBlock, 128, 256, 1, stride=2)
        # Layer 4
        self.layer4 = self._make_layer(BasicBlock, 256, 512, 1, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        return self

    def _make_layer(self, block, in_planes, planes, num_blocks, stride):
        layers = []
        layers.append(block(in_planes, planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

