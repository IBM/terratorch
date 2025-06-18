import importlib
import sys
from collections.abc import Callable
import logging 

import timm
import torch
from torch import nn

import terratorch.models.decoders as decoder_registry
from terratorch.models.backbones.clay_v1.embedder import Embedder
from terratorch.models.model import (
    AuxiliaryHead,
    AuxiliaryHeadWithDecoderWithoutInstantiatedHead,
    Model,
    ModelFactory,
)
from terratorch.models.pixel_wise_model import PixelWiseModel
from terratorch.models.scalar_output_model import ScalarOutputModel
from terratorch.models.utils import DecoderNotFoundError, extract_prefix_keys
from terratorch.registry import MODEL_FACTORY_REGISTRY
from claymodel.model import ClayMAE

PIXEL_WISE_TASKS = ["segmentation", "regression"]
SCALAR_TASKS = ["classification"]
SUPPORTED_TASKS = PIXEL_WISE_TASKS + SCALAR_TASKS


class DecoderNotFoundError(Exception):
    pass

class ModelWrapper(nn.Module):

    def __init__(self, model: nn.Module = None) -> None:

        super(ModelWrapper, self).__init__()

        self.model = model

        #self.embedding_shape = self.model.state_dict()['decoder.embed_to_pixels.dem.bias'].shape[0]

    def channels(self):
        return (1, self.embedding_shape)

    @property
    def parameters(self):
        return self.model.parameters

    def forward(self, args, **kwargs):
        datacube = {}
        datacube['pixels'] = args
        datacube['timestep'] = None
        datacube['platform'] = ['sentinel-2-l2a']
        datacube['latlon'] = None
        return_value = self.model.forward(args, **kwargs)
        print('r----------------------------------')
        print(return_value.keys())
        print('----------------------------------')
        return_value['image'] = return_value['pixels']
        return return_value

@MODEL_FACTORY_REGISTRY.register
class Clay1_5ModelFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: str | nn.Module,
        in_channels: int,
        bands: list[int] = [],
        num_classes: int | None = None,
        pretrained: bool = True,  # noqa: FBT001, FBT002
        num_frames: int = 1,
        prepare_features_for_image_model: Callable | None = None,
        aux_decoders: list[AuxiliaryHead] | None = None,
        rescale: bool = True,  # noqa: FBT002, FBT001
        checkpoint_path: str = None,
        **kwargs,
    ) -> Model:
        return ModelWrapper(ClayMAE(**kwargs))
