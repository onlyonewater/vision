import sys
import warnings
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception as inception_module
from torch import Tensor
from torch.jit.annotations import Optional
from torchvision.models.utils import load_state_dict_from_url


__all__ = [
    "QuantizableInception3",
    "inception_v3",
    "QuantizableInceptionOutputs",
    "_QuantizableInceptionOutputs",
]


model_urls = {
    # Inception v3 ported from TensorFlow
    "inception_v3_google": "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth"
}

QuantizableInceptionOutputs = namedtuple(
    "QuantizableInceptionOutputs", ["logits", "aux_logits"]
)
QuantizableInceptionOutputs.__annotations__ = {
    "logits": torch.Tensor,
    "aux_logits": Optional[torch.Tensor],
}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_QuantizableInceptionOutputs = QuantizableInceptionOutputs


def inception_v3(pretrained_float_model=False, progress=True, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        pretrained_float_model (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained_float_model:
        if "transform_input" not in kwargs:
            kwargs["transform_input"] = True
        if "aux_logits" in kwargs:
            original_aux_logits = kwargs["aux_logits"]
            kwargs["aux_logits"] = True
        else:
            original_aux_logits = False
        model = QuantizableInception3(**kwargs)
        state_dict = load_state_dict_from_url(
            model_urls["inception_v3_google"], progress=progress
        )
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.AuxLogits
        return model

    return QuantizableInception3(**kwargs)


class QuantizableBasicConv2d(inception_module.BasicConv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, ["conv", "bn", "relu"], inplace=True)


class QuantizableInceptionA(inception_module.InceptionA):
    def __init__(self, *args, **kwargs):
        super().__init__(basic_conv2d=QuantizableBasicConv2d, *args, **kwargs)
        self.myop = nn.quantized.FloatFunctional()

    def forward(self, x):
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInceptionB(inception_module.InceptionB):
    def __init__(self, *args, **kwargs):
        super().__init__(basic_conv2d=QuantizableBasicConv2d, *args, **kwargs)
        self.myop = nn.quantized.FloatFunctional()

    def forward(self, x):
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInceptionC(inception_module.InceptionC):
    def __init__(self, *args, **kwargs):
        super().__init__(basic_conv2d=QuantizableBasicConv2d, *args, **kwargs)
        self.myop = nn.quantized.FloatFunctional()

    def forward(self, x):
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInceptionD(inception_module.InceptionD):
    def __init__(self, *args, **kwargs):
        super().__init__(basic_conv2d=QuantizableBasicConv2d, *args, **kwargs)
        self.myop = nn.quantized.FloatFunctional()

    def forward(self, x):
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInceptionE(inception_module.InceptionE):
    def __init__(self, *args, **kwargs):
        super().__init__(basic_conv2d=QuantizableBasicConv2d, *args, **kwargs)
        self.myop = nn.quantized.FloatFunctional()

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = self.myop.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = self.myop.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInception3(inception_module.Inception3):
    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        super().__init__(
            num_classes=num_classes,
            aux_logits=aux_logits,
            transform_input=transform_input,
            basic_conv2d=QuantizableBasicConv2d,
            inception_a=QuantizableInceptionA,
            inception_b=QuantizableInceptionB,
            inception_c=QuantizableInceptionC,
            inception_d=QuantizableInceptionD,
            inception_e=QuantizableInceptionE,
        )
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self._transform_input(x)
        x = self.quant(x)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux = self.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        x = self.dequant(x)
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted QuantizableInception3 always returns QuantizableInception3 Tuple")
            return QuantizableInceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in googlenet model

        Fuse conv+bn+relu/ conv+relu/conv+bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        for m in self.modules():
            if type(m) == QuantizableBasicConv2d:
                torch.quantization.fuse_modules(m, ["conv", "bn", "relu"], inplace=True)
