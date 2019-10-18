import warnings
from collections import namedtuple
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import torchvision.models.googlenet
import sys
from torch import Tensor
from torch.jit.annotations import Optional

googlenet_module = sys.modules['torchvision.models.googlenet']

__all__ = ['QuantizableGoogLeNet', 'googlenet', "QuantizableGoogLeNetOutputs", "_QuantizableGoogLeNetOutputs"]

model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

QuantizableGoogLeNetOutputs = namedtuple('QuantizableGoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
QuantizableGoogLeNetOutputs.__annotations__ = {'logits': Tensor, 'aux_logits2': Optional[Tensor],
                                    'aux_logits1': Optional[Tensor]}

# Script annotations failed with _QuantizableGoogleNetOutputs = namedtuple ...
# _QuantizableGoogLeNetOutputs set here for backwards compat
_QuantizableGoogLeNetOutputs = QuantizableGoogLeNetOutputs


def googlenet(pretrained_float_model=False, progress=True, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.

    Args:
        pretrained_float_model (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained_float_model is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained_float_model:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' not in kwargs:
            kwargs['aux_logits'] = False
        if kwargs['aux_logits']:
            warnings.warn('auxiliary heads in the pretrained_float_model googlenet model are NOT pretrained_float_model, '
                          'so make sure to train them')
        original_aux_logits = kwargs['aux_logits']
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False
        model = QuantizableGoogLeNet(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['googlenet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.aux1, model.aux2
        return model

    return QuantizableGoogLeNet(**kwargs)


class QuantizableBasicConv2d(googlenet_module.BasicConv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class QuantizableInception(googlenet_module.Inception):

    def __init__(self, *args, **kwargs):
        super().__init__(BasicConv2d=QuantizableBasicConv2d, *args, **kwargs)
        self.cat = nn.quantized.FloatFunctional()

    def forward(self, x):
        outputs = self._forward(x)
        return self.cat.cat(outputs, 1)


class QuantizableGoogLeNet(googlenet_module.GoogLeNet):

    def __init__(self, *args, **kwargs):
        super().__init__(
            BasicConv2d=QuantizableBasicConv2d,
            Inception=QuantizableInception,
            *args,
            **kwargs
        )
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self._transform_input(x)
        x = self.quant(x)
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux1 = self.aux1(x)
        else:
            aux1 = None

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if aux_defined:
            aux2 = self.aux2(x)
        else:
            aux2 = None

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        x = self.dequant(x)
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted QuantizableGoogleNet always returns QuantizableGoogleNetOutputs Tuple")
            return QuantizableGoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in googlenet model

        Fuse conv+bn+relu/ conv+relu/conv+bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        for m in self.modules():
            if type(m) == QuantizableBasicConv2d:
                torch.quantization.fuse_modules(m, ["conv", "bn", "relu"], inplace=True)
