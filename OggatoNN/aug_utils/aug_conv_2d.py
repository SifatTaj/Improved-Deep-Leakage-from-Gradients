import collections
import torch
from itertools import repeat
from typing import Optional, List, Tuple, Union, TypeVar

from torch import Tensor
import torch.nn.functional as F
from torch._torch_docs import reproducibility_notes
from torch.nn.modules.conv import _ConvNd, convolution_notes

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import math

def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_pair = _ntuple(2, "_pair")
T = TypeVar('T')
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]


class AnonConv2d(_ConvNd):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            aug_indices=None,
            deanon_dim=None,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        self.deanon_dim = deanon_dim
        self.aug_indices = aug_indices
        super(AnonConv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:

        # img = input[0]
        # plt.imshow(img.cpu().permute(1, 2, 0))
        # plt.show()

        if (input.shape == (input.shape[0], input.shape[1], self.deanon_dim, self.deanon_dim)):
            # print('original data:', input.shape)
            return self._conv_forward(input, self.weight, self.bias)

        deanon_batch = torch.zeros(input.shape[0], input.shape[1], self.deanon_dim, self.deanon_dim)

        for idx, data in enumerate(input):
            deanon_img = torch.zeros(data.shape[0], self.deanon_dim, self.deanon_dim)
            try:
                aug_img_np = data.cpu().numpy()
            except:
                aug_img_np = data.cpu()
                aug_img_np = aug_img_np.detach().numpy()

            aug_img_np = aug_img_np.reshape(aug_img_np.shape[0], aug_img_np.shape[1] * aug_img_np.shape[1])

            for c in range(data.shape[0]):
                deanon_img_np = np.delete(aug_img_np[c], self.aug_indices[c])
                deanon_img_np = deanon_img_np.reshape(self.deanon_dim, self.deanon_dim)
                deanon_img[c] = torch.from_numpy(deanon_img_np)

            deanon_batch[idx] = deanon_img
        deanon_batch = deanon_batch.to('cuda')

        # test deanon
        # img = deanon_batch[0]
        # plt.imshow(img.cpu().permute(1, 2, 0))
        # plt.show()

        # print('deanon data:', deanon_batch.shape)

        return self._conv_forward(deanon_batch, self.weight, self.bias)
