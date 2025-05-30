import types

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    HyperLatentCodec,
    HyperpriorLatentCodec,
)
from compressai.layers import (
    AttentionBlock,
    CheckerboardMaskedConv2d,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv1x1,
    conv3x3,
    sequential_channel_ramp,
    subpel_conv3x3,
)
from compressai.registry import register_model

from base import SimpleVAECompressionModel
from utils1 import conv, deconv


class Elic2022ChandelierLite(SimpleVAECompressionModel):
    """轻量版ELIC 2022模型，移除了注意力模块并减少了残差块数量"""

    def __init__(self, N=192, M=160, groups=None, **kwargs):
        super().__init__(**kwargs)

        if groups is None:
            groups = [16, 16, 32, 32, M - 96]

        self.groups = list(groups)
        assert sum(self.groups) == M

        # 简化后的编码器网络，移除了注意力模块，每组只保留1个残差块
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            conv(N, M, kernel_size=5, stride=2),
        )

        # 简化后的解码器网络
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        # 简化后的超先验网络
        h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
        )

        h_s = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N, N * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(N * 3 // 2, M * 2, kernel_size=3, stride=1),
        )

        # 简化后的通道上下文
        channel_context = {
            f"y{k}": nn.Sequential(
                conv(
                    self.groups[0] + (k > 1) * self.groups[k - 1],
                    128,  # 减少通道数
                    kernel_size=5,
                    stride=1,
                ),
                nn.ReLU(inplace=True),
                conv(128, 64, kernel_size=5, stride=1),  # 减少通道数
                nn.ReLU(inplace=True),
                conv(64, self.groups[k] * 2, kernel_size=5, stride=1),
            )
            for k in range(1, len(self.groups))
        }

        # 其余部分保持不变
        spatial_context = [
            CheckerboardMaskedConv2d(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        param_aggregation = [
            nn.Sequential(
                conv1x1(
                    self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + M * 2,
                    M * 2,
                ),
                nn.ReLU(inplace=True),
                conv1x1(M * 2, 256),  # 减少通道数
                nn.ReLU(inplace=True),
                conv1x1(256, self.groups[k] * 2),
            )
            for k in range(len(self.groups))
        ]

        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(
                        quantizer="ste", chunks=("means", "scales")
                    ),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        self.latent_codec = HyperpriorLatentCodec(
            latent_codec={
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )
        self._monkey_patch()

    def _monkey_patch(self):
        """Monkey-patch to use only first group and most recent group."""

        def merge_y(self: ChannelGroupsLatentCodec, *args):
            if len(args) == 0:
                return Tensor()
            if len(args) == 1:
                return args[0]
            if len(args) < len(self.groups):
                return torch.cat([args[0], args[-1]], dim=1)
            return torch.cat(args, dim=1)

        chan_groups_latent_codec = self.latent_codec["y"]
        obj = chan_groups_latent_codec
        obj.merge_y = types.MethodType(merge_y, obj)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net

class ResidualBottleneckBlock(nn.Module):
    """Residual bottleneck block.

    Introduced by [He2016], this block sandwiches a 3x3 convolution
    between two 1x1 convolutions which reduce and then restore the
    number of channels. This reduces the number of parameters required.

    [He2016]: `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/abs/1512.03385>`_, by Kaiming He, Xiangyu Zhang,
    Shaoqing Ren, and Jian Sun, CVPR 2016.

    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = min(in_ch, out_ch) // 2
        self.conv1 = conv1x1(in_ch, mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_ch, mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(mid_ch, out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)

        out = x
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        return out + identity
