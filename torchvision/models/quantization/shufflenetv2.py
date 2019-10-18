import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import torchvision.models.shufflenetv2
import sys
from .utils import _replace_relu

shufflenetv2 = sys.modules['torchvision.models.shufflenetv2']

__all__ = [
    'QuantizableShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
]

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


class QuantizableInvertedResidual(shufflenetv2.InvertedResidual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cat = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = self.cat.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = self.cat.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = shufflenetv2.channel_shuffle(out, 2)

        return out


class QuantizableShuffleNetV2(shufflenetv2.ShuffleNetV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, InvertedResidual=QuantizableInvertedResidual, **kwargs)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in shufflenetv2 model

        Fuse conv+bn+relu/ conv+relu/conv+bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        for name, m in self._modules.items():
            if name in ["conv1", "conv5"]:
                torch.quantization.fuse_modules(m, [["0", "1", "2"]], inplace=True)
        for m in self.modules():
            if type(m) == QuantizableInvertedResidual:
                if len(m.branch1._modules.items()) > 0:
                    torch.quantization.fuse_modules(
                        m.branch1, [["0", "1"], ["2", "3", "4"]], inplace=True
                    )
                torch.quantization.fuse_modules(
                    m.branch2,
                    [["0", "1", "2"], ["3", "4"], ["5", "6", "7"]],
                    inplace=True,
                )

def _shufflenetv2(arch, pretrained_float_model, progress, *args, **kwargs):
    model = QuantizableShuffleNetV2(*args, **kwargs)
    _replace_relu(model)

    if pretrained_float_model:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained_float_model {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)

    return model


def shufflenet_v2_x0_5(pretrained_float_model=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained_float_model (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x0.5', pretrained_float_model, progress,
                         [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(pretrained_float_model=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained_float_model (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.0', pretrained_float_model, progress,
                         [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


def shufflenet_v2_x1_5(pretrained_float_model=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained_float_model (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.5', pretrained_float_model, progress,
                         [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_x2_0(pretrained_float_model=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained_float_model (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x2.0', pretrained_float_model, progress,
                         [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)
