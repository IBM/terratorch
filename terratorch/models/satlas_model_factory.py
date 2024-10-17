# Copyright contributors to the Terratorch project

try:
    import satlaspretrain_models
except ImportError as e:
    msg = "satlaspretrain_models must be installed to use the SatlasModelFactory.\
        Please install first with `pip install satlaspretrain-models` or pip install terratorch[satlas]"
    raise ImportError(msg) from e

import warnings
from urllib.parse import urlparse

import huggingface_hub
import torch
import torch.nn
import torch.nn.functional as F  # noqa: N812
from satlaspretrain_models.utils import Backbone, SatlasPretrain_weights

from terratorch.datasets.utils import HLSBands
from terratorch.models.backbones.select_patch_embed_weights import create_appropriate_patch_embed_from_pretrained
from terratorch.models.model import (
    AuxiliaryHead,
    Model,
    ModelFactory,
    ModelOutput,
    register_factory,
)

PRETRAINED_BANDS_RGB = [HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE]

# not sure about this order
PRETRAINED_BANDS_MS = [
    HLSBands.RED,
    HLSBands.GREEN,
    HLSBands.BLUE,
    HLSBands.RED_EDGE_1,
    HLSBands.RED_EDGE_2,
    HLSBands.RED_EDGE_3,
    HLSBands.NIR_BROAD,
    HLSBands.SWIR_1,
    HLSBands.SWIR_2,
]

FINETUNED_MODELS_DICT = {
    "marine_infrastructure": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/finetuned_satlas_explorer_models_2023-07-24/finetuned_satlas_explorer_sentinel2_marine_infrastructure.pth?download=true",
        "backbone": Backbone.SWINB,
        "head": satlaspretrain_models.Head.DETECT,
        "num_channels": 3,
        "multi_image": False,
        "sub_directory": "finetuned_satlas_explorer_models_2023-07-24",
    },
    "solar_farm": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/finetuned_satlas_explorer_models_2023-07-24/finetuned_satlas_explorer_sentinel2_solar_farm.pth?download=true",
        "backbone": Backbone.SWINB,
        "head": satlaspretrain_models.Head.BINSEGMENT,
        "num_channels": 9,
        "multi_image": True,
        "sub_directory": "finetuned_satlas_explorer_models_2023-07-24",
    },
    "tree_cover": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/finetuned_satlas_explorer_models_2023-07-24/finetuned_satlas_explorer_sentinel2_tree_cover.pth?download=true",
        "backbone": Backbone.SWINB,
        "head": satlaspretrain_models.Head.REGRESS,
        "num_channels": 9,
        "multi_image": True,
        "sub_directory": "finetuned_satlas_explorer_models_2023-07-24",
    },
    "wind_turbine": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/finetuned_satlas_explorer_models_2023-07-24/finetuned_satlas_explorer_sentinel2_wind_turbine.pth?download=true",
        "backbone": Backbone.SWINB,
        "head": satlaspretrain_models.Head.DETECT,
        "num_channels": 9,
        "multi_image": True,
        "sub_directory": "finetuned_satlas_explorer_models_2023-07-24",
    },
}

def adapt_weights_for_satlas_pretrained_models(weights: dict) -> dict:
    new_weights = {}
    for name, val in weights.items():
        new_name = name
        if "intermediates" in name:
            if "fpn" in name:
                new_name = "fpn." + new_name[new_name.find("fpn"):]
            else:
                new_name = "upsample." + new_name[new_name.find("layers"):]
        elif "heads" in name:
            new_name = "head." + new_name[new_name.find("layers"):]

        new_weights[new_name] = val
    return new_weights

class SatlasModelWrapper(Model):
    def __init__(self, satlas_model: satlaspretrain_models.Model, rescale: bool = True):
        """Wrapper for Satlas Models created from `satlaspretrain_models` to be compatible with terratorch.

        Args:
            satlas_model (satlaspretrain_models.Model): Satlas Model.
            rescale (bool): Whether to apply bilinear interpolation to rescale the model output if its size
                is different from the ground truth. Defaults to True.
        """
        super().__init__()
        self.model = satlas_model
        self.rescale = rescale

    def __repr__(self):
        return repr(self.model)

    def __str__(self):
        return str(self.model)

    def freeze_encoder(self):
        for param in self.model.backbone.parameters():
            param.requires_grad_(False)
        return super().freeze_encoder()

    def freeze_decoder(self):
        if hasattr(self.model, "head"):
            for param in self.model.head.parameters():
                param.requires_grad_(False)

        if hasattr(self.model, "fpn"):
            for param in self.model.fpn.parameters():
                param.requires_grad_(False)

        if hasattr(self.model, "upscale"):
            for param in self.model.upscale.parameters():
                param.requires_grad_(False)
        return super().freeze_decoder()

    def forward(self, x, *args, **kwargs) -> ModelOutput:
        # returns a tuple with output and loss
        output, _ = self.model.forward(x, *args, **kwargs)
        # interpolate to patch image size
        if self.rescale and output.shape[-2:] != x.shape[-2:]:
            squeezed = False
            if len(output.shape) < 4:
                output = output.unsqueeze(1) # simulate channels dimension
                squeezed = True
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear")
            if squeezed:
                output = output.squeeze(1) # remove simulated channels dimension
        return ModelOutput(output)

@register_factory
class SatlasModelFactory(ModelFactory):
    def __init__(self):
        # add finetuned models
        self.weights_dict = SatlasPretrain_weights.copy() | FINETUNED_MODELS_DICT

    def build_model(
        self,
        task: str,
        model_identifier: str,
        bands: list[HLSBands | int],
        aux_decoders: list[AuxiliaryHead] | None = None,
        pretrained: bool | str = True,
        pretrained_task_idx_head: int | None = None,
        load_upsample_weights: bool = True,
        fpn: bool = True,
        num_classes: int | None = None,
        rescale: bool = True
    ) -> SatlasModelWrapper:
        """Model factory for Satlas models. Adapted from https://github.com/allenai/satlaspretrain_models/blob/main/satlaspretrain_models/model.py

        This factory will create models using the `satlaspretrain_models` library,
        which has been created by the authors of SatlasNet.
        As such, this needs dependency to be installed in order to use this factory.
        Please install first with `pip install satlaspretrain-models` or `pip install terratorch[satlas]`.

        Args:
            task (str): Task to be performed. Currently supports "segmentation", "regression" and "classification".
            model_identifier (str): Model identifier to be passed to satlaspretrain_models.
                See that library's documentation for details.
            bands (list[terratorch.datasets.HLSBands | int]): Bands the model will be trained on.
                Should be a list of terratorch.datasets.HLSBands or ints.
            aux_decoders (list[AuxiliaryHead] | None, optional): Not supported by this factory.
            pretrained (bool | str, optional): Whether the model pretrained weights should be loaded.
                If a path, will load the weights from there. Defaults to True.
            pretrained_task_idx_head (int, optional): If pretrained is true, whether to load head weights from a certain task index.
                Defaults to None, which initializes a head with new weights.
            load_upsample_weights (bool): Whether to load the upsample weights as well. Defaults to True.
            fpn (bool, optional): Whether or not to feed imagery through the pretrained Feature Pyramid Network
                after the backbone. Defaults to True.
            num_classes (int, optional): Number of classes. May be 1 or None for regression tasks.
                Must be specified for other tasks. Defaults to None.
            rescale (bool): Whether to apply bilinear interpolation to rescale the model output if its size
                is different from the ground truth. Defaults to True.

        Returns:
            Model: Satlas model wrapped in a SatlasModelWrapper.
        """

        if aux_decoders:
            msg = "SatlasModelFactory does not support auxiliary decoders"
            raise NotImplementedError(msg)
        if model_identifier not in self.weights_dict.keys():
            msg = f"Invalid model_identifier. Must be one of {list(self.weights_dict.keys())}."
            raise ValueError(msg)

        bands = [HLSBands.try_convert_to_hls_bands_enum(b) for b in bands]

        if task == "regression":
            head = satlaspretrain_models.Head.REGRESS
        elif task == "segmentation":
            head = satlaspretrain_models.Head.SEGMENT
        elif task == "classification":
            head = satlaspretrain_models.Head.CLASSIFY
        else:
            msg = "Currently only supports regression, segmentation and classification."
            raise NotImplementedError(msg)

        if head and (num_classes is None):
            if head is satlaspretrain_models.Head.REGRESS:
                num_classes = 1
            else:
                msg = "Must specify num_classes if head is desired."
                raise ValueError(msg)

        model_info = self.weights_dict[model_identifier]
        in_channels = len(bands)
        if in_channels != model_info["num_channels"]:
            warnings.warn(
                f"Overwriting number of input channels from default \
                        of {model_info['num_channels']} to {in_channels}",
                stacklevel=1,
            )

        if "head" in model_info:
            if head:
                warnings.warn(
                    f"head was specified, but this model already specifies a head.\
                    Overwriting {model_info['head']} with {head}",
                    stacklevel=1,
                )
            else:
                head = model_info["head"]

        if head and (num_classes is None):
            if head is satlaspretrain_models.Head.REGRESS:
                num_classes = 1
            else:
                msg = "Must specify num_classes if head is desired."
                raise ValueError(msg)

        if pretrained:
            if isinstance(pretrained, bool):
                weights_url = model_info["url"]
                parsed_url = urlparse(weights_url)

                # Extract the file name from the path
                file_name = parsed_url.path.split("/")[-1]
                if "sub_directory" in model_info:
                    weights_file = huggingface_hub.hf_hub_download(
                        repo_id="allenai/satlas-pretrain", filename=file_name, subfolder=model_info["sub_directory"]
                    )
                else:
                    weights_file = huggingface_hub.hf_hub_download(
                        repo_id="allenai/satlas-pretrain", filename=file_name
                    )
            else:
                weights_file = pretrained

            weights = torch.load(weights_file, map_location="cpu")
        else:
            weights = None

        if model_info["num_channels"] == 9:
            pretrained_bands = PRETRAINED_BANDS_MS
        elif model_info["num_channels"] == 3:
            pretrained_bands = PRETRAINED_BANDS_RGB
        else:
            msg = f"Model number of channels {model_info['num_channels']} not known from pretraining"
            raise Exception(msg)

        if model_identifier not in FINETUNED_MODELS_DICT:
            # Adjust the first conv / patch embed layer for the appropriate number of channels
            if model_info["backbone"] in [Backbone.SWINB, Backbone.SWINT]:
                patch_embed_weights_name = "backbone.backbone.features.0.0.weight"
                pretrained_patch_embed_weights = weights[patch_embed_weights_name]
                patch_embed_out_channels = pretrained_patch_embed_weights.shape[0]
                temp_weight = torch.nn.Conv2d(
                    in_channels, patch_embed_out_channels, kernel_size=(4, 4), stride=(4, 4)
                ).state_dict()["weight"]
                create_appropriate_patch_embed_from_pretrained(
                    temp_weight, pretrained_patch_embed_weights, pretrained_bands, bands
                )
                weights[patch_embed_weights_name] = temp_weight
            elif model_info["backbone"] in [Backbone.RESNET152, Backbone.RESNET50]:
                patch_embed_weights_name = "backbone.resnet.conv1.weight"
                pretrained_patch_embed_weights = weights[patch_embed_weights_name]
                patch_embed_out_channels = pretrained_patch_embed_weights.shape[0]
                temp_weight = torch.nn.Conv2d(
                    in_channels, patch_embed_out_channels, kernel_size=7, stride=2, padding=3, bias=False
                ).state_dict()["weight"]
                create_appropriate_patch_embed_from_pretrained(
                    temp_weight, pretrained_patch_embed_weights, pretrained_bands, bands
                )
                weights[patch_embed_weights_name] = temp_weight
            else:
                warnings.warn("Backbone type not Swin or ResNet. Patch embed weights may not match\
                            or be aligned with input data channels", stacklevel=1)
            # Initialize a pretrained model using the Model() class.
            model = satlaspretrain_models.Model(
                in_channels,
                model_info["multi_image"],
                model_info["backbone"],
                fpn=fpn,
                head=head,
                num_categories=num_classes,
                weights=weights)
            if load_upsample_weights:
                appropriate_weights = {k.replace(f"intermediates.1.", "", 1): v for k, v in weights.items() if f"intermediates.1." in k}
                model.upsample.load_state_dict(appropriate_weights)
            if pretrained_task_idx_head:
                appropriate_weights = {k.replace(f"heads.{pretrained_task_idx_head}.", "", 1): v for k, v in weights.items() if f"heads.{pretrained_task_idx_head}." in k}
                model.head.load_state_dict(appropriate_weights)

        else:
            # for finetuned models, weights must be loaded like this
            model = satlaspretrain_models.Model(
                in_channels,
                model_info["multi_image"],
                model_info["backbone"],
                fpn=fpn,
                head=head,
                num_categories=num_classes,
                weights=None)
            weights = adapt_weights_for_satlas_pretrained_models(weights)
            model.load_state_dict(weights)

        return SatlasModelWrapper(model, rescale=rescale and task in ["segmentation", "regression"])
