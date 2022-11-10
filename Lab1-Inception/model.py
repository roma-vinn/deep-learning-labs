from typing import Optional

from buildingblocks import InceptionConfig, InceptionBlock, BasicConv2d
from torch import nn, Tensor
import torch
from torchvision import transforms


inception_config_dict = {
    '3a': InceptionConfig(num_1x1=64, num_3x3_reduce=96, num_3x3=128, num_5x5_reduce=16, num_5x5=32, pool_proj=32),
    '3b': InceptionConfig(num_1x1=128, num_3x3_reduce=128, num_3x3=192, num_5x5_reduce=32, num_5x5=96, pool_proj=64),

    '4a': InceptionConfig(num_1x1=192, num_3x3_reduce=96, num_3x3=208, num_5x5_reduce=16, num_5x5=48, pool_proj=64),
    '4b': InceptionConfig(num_1x1=160, num_3x3_reduce=112, num_3x3=224, num_5x5_reduce=24, num_5x5=64, pool_proj=64),
    '4c': InceptionConfig(num_1x1=128, num_3x3_reduce=128, num_3x3=256, num_5x5_reduce=24, num_5x5=64, pool_proj=64),
    '4d': InceptionConfig(num_1x1=112, num_3x3_reduce=144, num_3x3=288, num_5x5_reduce=32, num_5x5=64, pool_proj=64),
    '4e': InceptionConfig(num_1x1=256, num_3x3_reduce=160, num_3x3=320, num_5x5_reduce=32, num_5x5=128, pool_proj=128),

    '5a': InceptionConfig(num_1x1=256, num_3x3_reduce=160, num_3x3=320, num_5x5_reduce=32, num_5x5=128, pool_proj=128),
    '5b': InceptionConfig(num_1x1=384, num_3x3_reduce=192, num_3x3=384, num_5x5_reduce=48, num_5x5=128, pool_proj=128),
}


class InceptionModel(nn.Module):
    def __init__(
            self,
            num_classes: int = 47,
            transform_input: bool = False,
            config_dict: Optional[dict] = None,
            dropout: float = 0.4,
    ) -> None:
        super(InceptionModel, self).__init__()
        if config_dict is None:
            config_dict = inception_config_dict
        self.transform_input = transform_input
        self._transforms = transforms.Normalize((0.1307,), (0.3081,))

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = BasicConv2d(1, 64, kernel_size=7, stride=1, padding=3)  # original: stride=2

        self.conv2_1 = BasicConv2d(64, 64, kernel_size=1)
        self.conv2_2 = BasicConv2d(64, 192, kernel_size=3, padding=1)

        self.inception3a = InceptionBlock(in_channels=192, config=config_dict['3a'])
        self.inception3b = InceptionBlock(in_channels=256, config=config_dict['3b'])

        self.inception4a = InceptionBlock(in_channels=480, config=config_dict['4a'])
        self.inception4b = InceptionBlock(in_channels=512, config=config_dict['4b'])
        self.inception4c = InceptionBlock(in_channels=512, config=config_dict['4c'])
        self.inception4d = InceptionBlock(in_channels=512, config=config_dict['4d'])
        self.inception4e = InceptionBlock(in_channels=528, config=config_dict['4e'])

        self.inception5a = InceptionBlock(in_channels=832, config=config_dict['5a'])
        self.inception5b = InceptionBlock(in_channels=832, config=config_dict['5b'])

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x = self._transforms(x)
        return x

    def _forward(self, x: Tensor) -> Tensor:
        # N x 1 x 28 x 28
        x = self.conv1(x)
        # x = self.max_pool(x)
        # N x 64 x 28 x 28
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        # x = self.max_pool(x)
        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.max_pool(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.max_pool(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7
        x = self.avg_pool(x)
        # N x 1024 x 1 x 1
        x = self.dropout(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.fc(x)
        # N x num_classes (47)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._transform_input(x)
        x = self._forward(x)
        return x  # logits


inception_small_config_dict = {
    '2a': InceptionConfig(num_1x1=32, num_3x3_reduce=48, num_3x3=64, num_5x5_reduce=8, num_5x5=16, pool_proj=16),

    '3a': InceptionConfig(num_1x1=32, num_3x3_reduce=64, num_3x3=96, num_5x5_reduce=16, num_5x5=32, pool_proj=32),
    '3b': InceptionConfig(num_1x1=64, num_3x3_reduce=96, num_3x3=128, num_5x5_reduce=16, num_5x5=32, pool_proj=32),

    '4a': InceptionConfig(num_1x1=96, num_3x3_reduce=128, num_3x3=160, num_5x5_reduce=32, num_5x5=64, pool_proj=64),
}


class InceptionModelSmall(nn.Module):
    def __init__(
            self,
            num_classes: int = 47,
            transform_input: bool = False,
            config_dict: Optional[dict] = None,
            dropout: float = 0.4,
    ) -> None:
        super(InceptionModelSmall, self).__init__()
        if config_dict is None:
            config_dict = inception_small_config_dict
        self.transform_input = transform_input
        self._transforms = transforms.Normalize((0.1307,), (0.3081,))

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = BasicConv2d(1, 64, kernel_size=7, stride=1, padding=3)

        self.inception2a = InceptionBlock(in_channels=64, config=config_dict['2a'])

        self.inception3a = InceptionBlock(in_channels=128, config=config_dict['3a'])
        self.inception3b = InceptionBlock(in_channels=192, config=config_dict['3b'])

        self.inception4a = InceptionBlock(in_channels=256, config=config_dict['4a'])

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(384, num_classes)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x = self._transforms(x)
        return x

    def _forward(self, x: Tensor) -> Tensor:
        # N x 1 x 28 x 28
        x = self.conv1(x)
        # N x 64 x 28 x 28
        x = self.inception2a(x)
        # N x 128 x 28 x 28
        x = self.max_pool(x)
        # N x 128 x 14 x 14
        x = self.inception3a(x)
        # N x 192 x 14 x 14
        x = self.inception3b(x)
        # N x 256 x 14 x 14
        x = self.max_pool(x)
        # N x 256 x 7 x 7
        x = self.inception4a(x)
        # N x 384 x 7 x 7
        x = self.avg_pool(x)
        # N x 384 x 1 x 1
        x = self.dropout(x)
        # N x 384 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 384
        x = self.fc(x)
        # N x num_classes (47)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._transform_input(x)
        x = self._forward(x)
        return x  # logits


def main():
    from dataloaders import get_dataloader

    # prepare test data
    test_loader = iter(get_dataloader(train=False, augment=False, batch_size=2))
    test_data, test_label = next(test_loader)
    print(test_data.shape)

    model = InceptionModel()
    test_output = model(test_data)
    print(test_output.shape)

    model_small = InceptionModelSmall()
    test_small_output = model_small(test_data)
    print(test_small_output.shape)


if __name__ == '__main__':
    main()
