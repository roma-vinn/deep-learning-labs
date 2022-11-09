from typing import Any, Optional, Callable, List

from dataclasses import dataclass

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


@dataclass
class InceptionConfig:
    """Class for storing configuration of specific Inception block with dimension reductions
     e.g. number of filters in each conv/pool branch"""
    num_1x1: int
    num_3x3_reduce: int
    num_3x3: int
    num_5x5_reduce: int
    num_5x5: int
    pool_proj: int


class InceptionBlock(nn.Module):
    """Inception module with dimension reductions according to https://arxiv.org/abs/1409.4842"""
    def __init__(
            self,
            in_channels: int,
            config: InceptionConfig,
            conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionBlock, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch_1x1 = conv_block(in_channels, config.num_1x1, kernel_size=1)

        self.branch_3x3_1 = conv_block(in_channels, config.num_3x3_reduce, kernel_size=1)
        self.branch_3x3_2 = conv_block(config.num_3x3_reduce, config.num_3x3, kernel_size=3, padding=1)

        self.branch_5x5_1 = conv_block(in_channels, config.num_5x5_reduce, kernel_size=1)
        self.branch_5x5_2 = conv_block(config.num_5x5_reduce, config.num_5x5, kernel_size=5, padding=2)

        self.branch_pool_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_2 = conv_block(in_channels, config.pool_proj, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch_1x1 = self.branch_1x1(x)

        branch_3x3 = self.branch_3x3_1(x)
        branch_3x3 = self.branch_3x3_2(branch_3x3)

        branch_5x5 = self.branch_5x5_1(x)
        branch_5x5 = self.branch_5x5_2(branch_5x5)

        branch_pool = self.branch_pool_1(x)
        branch_pool = self.branch_pool_2(branch_pool)

        outputs = [branch_1x1, branch_3x3, branch_5x5, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)
