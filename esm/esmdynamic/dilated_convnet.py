"""
Dilated Convolutional Neural Network for RMSD prediction.
"""

from dataclasses import dataclass

import torch.nn as nn

from .utils import rmsd_vals


class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=1, padding='same'),
            nn.BatchNorm1d(out_channels))
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
class DilatedConvNetConfig:
    in_channels: int = 1024
    dim_reduction_layers: tuple = (512, 256, 128, 64, 32)
    dilations: tuple = (1, 2, 4, 8, 16, 32, 64, 128)
    kernel_size: int = 7
    num_classes: int = len(rmsd_vals)


class DilatedConvNet(nn.Module):
    """
    A dilated convolutional NN for one dimensional input.
    """

    def __init__(self, **kwargs):
        """
        Inputs:
            config: dataclass encapsulating parameters.
                in_channels: int = 1024
                dim_reduction_layers: tuple = (512, 256, 128, 64, 32)
                dilations: tuple = (1, 2, 4, 8, 16, 32, 64, 128)
                block: callable = DilatedResidualBlock
                kernel_size: int = 7
                num_classes: int = len(rmsd_bin_boundaries) --> We defined 32 bins
        """
        super(DilatedConvNet, self).__init__()
        self.cfg = DilatedConvNetConfig(**kwargs)
        in_channels = self.cfg.in_channels
        dim_reduction_layers = self.cfg.dim_reduction_layers
        dilations = self.cfg.dilations
        block = DilatedResidualBlock
        kernel_size = self.cfg.kernel_size  # Kernel size for dilated convolution modules
        num_classes = self.cfg.num_classes  # "Classes" correspond to bins for RMSD (see utils.py)

        self.dim_reduction_modules = nn.ModuleList()
        for out_channels in dim_reduction_layers:
            self.dim_reduction_modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='valid'),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU()
                )
            )
            in_channels = out_channels

        out_channels = dim_reduction_layers[-1]  # Final number of channels
        self.dilated_conv_modules = nn.ModuleList()
        for dilation in dilations:
            self.dilated_conv_modules.append(
                block(out_channels, out_channels, kernel_size, dilation)
            )

        self.prediction_layer = nn.Conv1d(out_channels, num_classes, kernel_size=1, padding='valid')

        # assert (num_classes >= 1)
        # if num_classes > 2:
        #     self.output_activation = nn.Softmax(dim=1)
        # elif num_classes <= 2:
        #     self.cfg.num_classes = 1
        #     self.output_activation = nn.Sigmoid()  # Eliminate redundant class

    def forward(self, x):
        for layer in self.dim_reduction_modules:
            x = layer(x)
        for layer in self.dilated_conv_modules:
            x = layer(x)
        x = self.prediction_layer(x)
        # x = self.output_activation(x) --> Computed afterwards
        return x
