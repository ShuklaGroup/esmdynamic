"""
ResNet for dynamic contact prediction.
"""

from dataclasses import dataclass
import typing as T

import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Adapted from https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


@dataclass
class ResNetConfig:
    in_channels: int = 128
    layer_dimensions: tuple = (64, 32, 16, 8, 4)
    res_block_num: int = 1  # Number of residual blocks between dimensionality reduction layers
    kernel_size: int = 5
    num_classes: int = 1


def _make_layer(block, input_channels, output_channels, kernel_size=5, layers_per_block=1):
    layers = []
    for i in range(layers_per_block):
        layers.append(block(input_channels, output_channels, kernel_size))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """
    ResNet for conditional probability prediction.
    """

    def __init__(self, cfg=None, **kwargs):
        """
        Inputs:
            config: dataclass encapsulating parameters.
                in_channels: int = 128
                layer_dimensions: tuple = (64, 32, 16, 8, 4)
                res_block_num: int = 1  # Number of residual blocks between dimensionality reduction layers
                kernel_size: int = 5
                num_classes: int = 1
        """
        super(ResNet, self).__init__()

        self.cfg = cfg if (cfg is not None) else ResNetConfig(**kwargs)
        in_channels = self.cfg.in_channels
        layer_dimensions = self.cfg.layer_dimensions
        res_block_num = self.cfg.res_block_num
        block = ResidualBlock
        kernel_size = self.cfg.kernel_size
        num_classes = self.cfg.num_classes

        self.layers = nn.ModuleList()
        for out_channels in layer_dimensions:
            self.layers.append(
                _make_layer(
                    block,
                    in_channels,
                    in_channels,
                    kernel_size,
                    res_block_num
                )
            )
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            )
            in_channels = out_channels

        out_channels = layer_dimensions[-1]
        self.prediction_layer = nn.Conv2d(out_channels, num_classes, kernel_size=1, padding='same')

        # assert (num_classes >= 1)
        # if num_classes > 2:
        #     self.output_activation = nn.Softmax(dim=-1)
        # elif num_classes <= 2:
        #     self.cfg.num_classes = 1
        #     self.output_activation = nn.Sigmoid()  # Eliminate redundant class

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.prediction_layer(x)
        # x = self.output_activation(x) --> Computed afterwards

        return x


class SymmetricResNet(ResNet):
    """
    ResNet (with Symmetric output) for dynamic contact prediction.
    """

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.prediction_layer(x)

        x += x.clone().mT  # Enforce symmetric output
        x /= 2  # Normalize from previous addition
        # x = self.output_activation(x) --> Computed afterwards

        return x
