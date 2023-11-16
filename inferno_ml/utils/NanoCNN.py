import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            Swish(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MicroCNN(nn.Module):

    def __init__(self, block_count, depth, width, residual, num_classes):
        super(MicroCNN, self).__init__()
        self.depth = depth
        self.width = width
        self.block_count = block_count
        self.num_classes = num_classes

        self.init_block = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=3, out_channels=64,
                      stride=3, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # 56*56
        self.blocks = nn.ModuleList()

        if residual == 0: 
            if self.block_count >= 1:
                self.blocks.append(nn.Sequential(
                    *(BottleneckResidualBlock(in_channels=(64 if d == 0 else int(64 * width)),
                                    out_channels=int(64 * width),
                                    downsample=False, first_conv=(d == 0)) for d in range(self.depth))
                ))
            if self.block_count >= 2:
                self.blocks.append(nn.Sequential(
                    *(BottleneckResidualBlock(in_channels=(64 * width if d == 0 else int(128 * width)),
                                    out_channels=int(128 * width),
                                    downsample=(d == 0), first_conv=(d == 0)) for d in range(self.depth))
                ))
            if self.block_count >= 3:
                self.blocks.append(nn.Sequential(
                    *(BottleneckResidualBlock(in_channels=(int(128 * width) if d == 0 else int(256 * width)),
                                    out_channels=int(256 * width),
                                    downsample=(d == 0), first_conv=(d == 0)) for d in range(self.depth))
                ))
            if self.block_count >= 4:
                self.blocks.append(nn.Sequential(
                    *(BottleneckResidualBlock(in_channels=(int(256 * width) if d == 0 else int(512 * width)),
                                    out_channels=int(512 * width),
                                    downsample=(d == 0), first_conv=(d == 0)) for d in range(self.depth))
                ))
            if self.block_count >= 5:
                self.blocks.append(nn.Sequential(
                    *(BottleneckResidualBlock(in_channels=(int(512 * width) if d == 0 else int(1024 * width)),
                                    out_channels=int(1024 * width),
                                    downsample=(d == 0), first_conv=(d == 0)) for d in range(self.depth))
                ))
            if self.block_count >= 6:
                self.blocks.append(nn.Sequential(
                    *(BottleneckResidualBlock(in_channels=(int(1024 * width) if d == 0 else int(2048 * width)),
                                    out_channels=int(2048 * width),
                                    downsample=(d == 0), first_conv=(d == 0)) for d in range(self.depth))
                ))
            if self.block_count >= 7:
                self.blocks.append(nn.Sequential(
                    *(BottleneckResidualBlock(in_channels=(int(2048 * width) if d == 0 else int(4096 * width)),
                                    out_channels=int(4096 * width),
                                    downsample=(d == 0), first_conv=(d == 0)) for d in range(self.depth))
                ))
            if self.block_count >= 8:
                raise ValueError('Wrong argument for B')
        else:
            if self.block_count >= 1:
                self.blocks.append(nn.Sequential(
                    *(ResidualBlock(in_channels=(64 if d == 0 else int(64 * width)),
                                    out_channels=int(64 * width),
                                    downsample=False, first_conv=(d == 0)) for d in range(self.depth))
                ))
            if self.block_count >= 2:
                self.blocks.append(nn.Sequential(
                    *(ResidualBlock(in_channels=(64 * width if d == 0 else int(128 * width)),
                                    out_channels=int(128 * width),
                                    downsample=(d == 0), first_conv=(d == 0)) for d in range(self.depth))
                ))
            if self.block_count >= 3:
                self.blocks.append(nn.Sequential(
                    *(ResidualBlock(in_channels=(int(128 * width) if d == 0 else int(256 * width)),
                                    out_channels=int(256 * width),
                                    downsample=(d == 0), first_conv=(d == 0)) for d in range(self.depth))
                ))
            if self.block_count >= 4:
                self.blocks.append(nn.Sequential(
                    *(ResidualBlock(in_channels=(int(256 * width) if d == 0 else int(512 * width)),
                                    out_channels=int(512 * width),
                                    downsample=(d == 0), first_conv=(d == 0)) for d in range(self.depth))
                ))
            if self.block_count >= 5:
                self.blocks.append(nn.Sequential(
                    *(ResidualBlock(in_channels=(int(512 * width) if d == 0 else int(1024 * width)),
                                    out_channels=int(1024 * width),
                                    downsample=(d == 0), first_conv=(d == 0)) for d in range(self.depth))
                ))
            if self.block_count >= 6:
                self.blocks.append(nn.Sequential(
                    *(ResidualBlock(in_channels=(int(1024 * width) if d == 0 else int(2048 * width)),
                                    out_channels=int(2048 * width),
                                    downsample=(d == 0), first_conv=(d == 0)) for d in range(self.depth))
                ))
            if self.block_count >= 7:
                self.blocks.append(nn.Sequential(
                    *(ResidualBlock(in_channels=(int(2048 * width) if d == 0 else int(4096 * width)),
                                    out_channels=int(4096 * width),
                                    downsample=(d == 0), first_conv=(d == 0)) for d in range(self.depth))
                ))
            if self.block_count >= 8:
                raise ValueError('Wrong argument for B')


        # Add the following line
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.last_layer = nn.Linear(in_features=int((32 * (2 ** self.block_count)) * width), out_features=num_classes)

    def forward(self, x):
        out = self.init_block(x)

        for i in range(self.block_count):
            out = self.blocks[i](out)

        out = F.adaptive_avg_pool2d(out, 1)    
        # out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.last_layer(out)
        return out

    def get_name(self):
        return "MicroCNN_B{}_W{}_D{}".format(self.block_count, self.width, self.depth)


class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, first_conv=False):
        super().__init__()

        mid_channels = out_channels // 4

        # First 1x1 convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=(2 if downsample and first_conv else 1),
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # Third 1x1 convolution
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if downsample and first_conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        elif first_conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

        self.activation = Swish()
        self.se = SEBlock(out_channels)

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # Apply Squeeze and Excitation block
        x = self.se(x)

        x = x + shortcut
        return self.activation(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, first_conv=False):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        if downsample and first_conv:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,
                                   padding=1, bias=False)

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        elif first_conv:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                                   padding=1, bias=False)

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + shortcut
        return self.relu(x)
