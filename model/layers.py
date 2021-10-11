import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 bias=False):
        super(ConvBNReLU, self).__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=(stride, stride), padding=kernel_size // 2, bias=bias)
        self._batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = F.relu(x)
        return x


class ConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 bias=False):
        super(ConvBN, self).__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=(stride, stride), padding=kernel_size // 2, bias=bias)
        self._batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x
