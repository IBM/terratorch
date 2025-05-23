# Copyright contributors to the Terratorch project

import torch
from typing import List
import warnings
import logging
from torch import nn

from terratorch.models.model import (
    AuxiliaryHead,
    AuxiliaryHeadWithDecoderWithoutInstantiatedHead,
    Model,
    ModelFactory,
)
from terratorch.models.necks import Neck, build_neck_list
from terratorch.models.peft_utils import get_peft_backbone
from terratorch.models.pixel_wise_model import PixelWiseModel
from terratorch.models.scalar_output_model import ScalarOutputModel
from terratorch.models.utils import extract_prefix_keys
from terratorch.registry import BACKBONE_REGISTRY, DECODER_REGISTRY, MODEL_FACTORY_REGISTRY

PIXEL_WISE_TASKS = ["segmentation", "regression"]
SCALAR_TASKS = ["classification"]
SUPPORTED_TASKS = PIXEL_WISE_TASKS + SCALAR_TASKS



class TemporalWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, pooling="mean", concat=False):
        """
        Wrapper for applying a temporal encoder across multiple time steps.

        Args:
            encoder (nn.Module): The feature extractor (backbone).
            pooling (str): Type of pooling ('mean' or 'max').
            concat (bool): Whether to concatenate features instead of pooling.
        """
        super().__init__()
        self.encoder = encoder
        self.concat = concat
        self.pooling_type = pooling

        if pooling not in ["mean", "max"]:
            raise ValueError("Pooling must be 'mean' or 'max'")

        # Ensure the encoder has an out_channels attribute
        if hasattr(encoder, "out_channels"):
            self.out_channels = encoder.out_channels * (1 if not concat else encoder.out_channels)
        else:
            raise AttributeError("Encoder must have an `out_channels` attribute.")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass for temporal processing.

        Args:
            x (Tensor): Input tensor of shape [B, C, T, H, W].

        Returns:
            List[Tensor]: A list of processed tensors, one per feature map.
        """
        if x.dim() != 5:
            raise ValueError(f"Expected input shape [B, C, T, H, W], but got {x.shape}")

        batch_size, channels, timesteps, height, width = x.shape

        # Initialize lists to store feature maps at each timestamp
        num_feature_maps = None  # Will determine dynamically
        features_per_map = []  # Stores feature maps across timestamps

        for t in range(timesteps):
            feat = self.encoder(x[:, :, t, :, :])  # Extract features at timestamp t

            if not isinstance(feat, list):  # If the encoder outputs a single feature map, convert to list
                if isinstance(feat, tuple):
                    feat = list(feat)
                else:
                    feat = [feat]

            if num_feature_maps is None:
                num_feature_maps = len(feat)  # Determine how many feature maps the encoder produces

            for i, feature_map in enumerate(feat):
                if len(features_per_map) <= i:
                    features_per_map.append([])  # Create list for each feature map
                    
                feature_map = feature_map[0] if isinstance(feature_map, tuple) else feature_map
                features_per_map[i].append(feature_map)  # Store feature map at time t

        # Stack features along the temporal dimension
        for i in range(num_feature_maps):
            try:
                features_per_map[i] = torch.stack(features_per_map[i], dim=2)  # Shape: [B, C', T, H', W']
            except RuntimeError as e:
                raise

        # Apply pooling or concatenation
        if self.concat:
            return [feat.view(batch_size, -1, height, width) for feat in features_per_map]  # Flatten T into C'
        elif self.pooling_type == "max":
            return [torch.max(feat, dim=2)[0] for feat in features_per_map]  # Max pooling across T
        else:
            return [torch.mean(feat, dim=2) for feat in features_per_map]  # Mean pooling across T

def _get_backbone(backbone: str | nn.Module, **backbone_kwargs) -> nn.Module:
    use_temporal = backbone_kwargs.pop('use_temporal', None)
    temporal_pooling = backbone_kwargs.pop('temporal_pooling', None)
    if isinstance(backbone, nn.Module):
        model = backbone
    else:
        model = BACKBONE_REGISTRY.build(backbone, **backbone_kwargs)

    # Apply TemporalWrapper inside _get_backbone
    if use_temporal:
        model = TemporalWrapper(model, pooling=temporal_pooling)

    return model


def _get_decoder_and_head_kwargs(
    decoder: str | nn.Module,
    channel_list: list[int],
    decoder_kwargs: dict,
    head_kwargs: dict,
    num_classes: int | None = None,
) -> tuple[nn.Module, dict, bool]:
    # if its already an nn Module, check if it includes a head. if it doesnt, pass num classes to head kwargs
    if isinstance(decoder, nn.Module):
        if not getattr(decoder, "includes_head", False) and num_classes is not None:
            head_kwargs["num_classes"] = num_classes
        elif head_kwargs:
            msg = "Decoder already includes a head, but `head_` arguments were specified. These should be removed."
            raise ValueError(msg)
        return decoder, head_kwargs, False

    # if its not an nn module, check if the class includes a head
    # depending on that, pass num classes to either head kwrags or decoder
    try:
        decoder_includes_head = DECODER_REGISTRY.find_class(decoder).includes_head
    except AttributeError as _:
        msg = (
            f"Decoder {decoder} does not have an `includes_head` attribute. Falling back to the value of the registry."
        )
        logging.warning(msg)
        decoder_includes_head = DECODER_REGISTRY.find_registry(decoder).includes_head
    if num_classes is not None:
        if decoder_includes_head:
            decoder_kwargs["num_classes"] = num_classes
            if head_kwargs:
                msg = "Decoder already includes a head, but `head_` arguments were specified. These should be removed."
                raise ValueError(msg)
        else:
            head_kwargs["num_classes"] = num_classes

    return DECODER_REGISTRY.build(decoder, channel_list, **decoder_kwargs), head_kwargs, decoder_includes_head


def _check_all_args_used(kwargs):
    if kwargs:
        msg = f"arguments {kwargs} were passed but not used."
        raise ValueError(msg)


def _get_argument_from_instance(model, name):
    return getattr(model._timm_module.patch_embed, name)[-1]


@MODEL_FACTORY_REGISTRY.register
class EncoderDecoderFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: str | nn.Module,
        backbone_kwargs: dict | None = None,
        decoder_kwargs: dict | None = None,
        head_kwargs: dict | None = None,
        num_classes: int | None = None,
        necks: list[dict] | None = None,
        aux_decoders: list[AuxiliaryHead] | None = None,
        rescale: bool = True,  # noqa: FBT002, FBT001,
        peft_config: dict | None = None,
        **kwargs,
    ) -> Model:
        """Generic model factory that combines an encoder and decoder, together with a head, for a specific task.

        Further arguments to be passed to the backbone, decoder or head. They should be prefixed with
        `backbone_`, `decoder_` and `head_` respectively.

        Args:
            task (str): Task to be performed. Currently supports "segmentation", "regression" and "classification".
            backbone (str, nn.Module): Backbone to be used. If a string, will look for such models in the different
                registries supported (internal terratorch registry, timm, ...). If a torch nn.Module, will use it
                directly. The backbone should have and `out_channels` attribute and its `forward` should return a list[Tensor].
            decoder (Union[str, nn.Module], optional): Decoder to be used for the segmentation model.
                    If a string, will look for such decoders in the different
                    registries supported (internal terratorch registry, smp, ...).
                    If an nn.Module, we expect it to expose a property `decoder.out_channels`.
                    Pixel wise tasks will be concatenated with a Conv2d for the final convolution.
                    Defaults to "FCNDecoder".
            backbone_kwargs (dict, optional) : Arguments to be passed to instantiate the backbone.
            decoder_kwargs (dict, optional) : Arguments to be passed to instantiate the decoder.
            head_kwargs (dict, optional) : Arguments to be passed to the head network. 
            num_classes (int, optional): Number of classes. None for regression tasks.
            necks (list[dict]): nn.Modules to be called in succession on encoder features
                before passing them to the decoder. Should be registered in the NECKS_REGISTRY registry.
                Expects each one to have a key "name" and subsequent keys for arguments, if any.
                Defaults to None, which applies the identity function.
            aux_decoders (list[AuxiliaryHead] | None): List of AuxiliaryHead decoders to be added to the model.
                These decoders take the input from the encoder as well.
            rescale (bool): Whether to apply bilinear interpolation to rescale the model output if its size
                is different from the ground truth. Only applicable to pixel wise models
                (e.g. segmentation, pixel wise regression). Defaults to True.
            peft_config (dict): Configuration options for using [PEFT](https://huggingface.co/docs/peft/index).
                The dictionary should have the following keys:

                - "method": Which PEFT method to use. Should be one implemented in PEFT, a list is available [here](https://huggingface.co/docs/peft/package_reference/peft_types#peft.PeftType).
                - "replace_qkv": String containing a substring of the name of the submodules to replace with QKVSep.
                  This should be used when the qkv matrices are merged together in a single linear layer and the PEFT
                  method should be applied separately to query, key and value matrices (e.g. if LoRA is only desired in
                  Q and V matrices). e.g. If using Prithvi this should be "qkv"
                - "peft_config_kwargs": Dictionary containing keyword arguments which will be passed to [PeftConfig](https://huggingface.co/docs/peft/package_reference/config#peft.PeftConfig)


        Returns:
            nn.Module: Full model with encoder, decoder and head.
        """
        task = task.lower()
        if task not in SUPPORTED_TASKS:
            msg = f"Task {task} not supported. Please choose one of {SUPPORTED_TASKS}"
            raise NotImplementedError(msg)

        if not backbone_kwargs:
            backbone_kwargs, kwargs = extract_prefix_keys(kwargs, "backbone_")

        backbone = _get_backbone(backbone, **backbone_kwargs)


        # If patch size is not provided in the config or by the model, it might lead to errors due to irregular images.
        patch_size = backbone_kwargs.get("patch_size", None)

        if patch_size is None:
            # Infer patch size from model by checking all backbone modules
            for module in backbone.modules():
                if hasattr(module, "patch_size"):
                    patch_size = module.patch_size
                    break
        padding = backbone_kwargs.get("padding", "reflect")

        if peft_config is not None:
            if not backbone_kwargs.get("pretrained", False):
                msg = (
                    "You are using PEFT without a pretrained backbone. If you are loading a checkpoint afterwards "
                    "this is probably fine, but if you are training a model check the backbone_pretrained parameter."
                )
                warnings.warn(msg, stacklevel=1)

            backbone = get_peft_backbone(peft_config, backbone)

        try:
            out_channels = backbone.out_channels
        except AttributeError as e:
            msg = "backbone must have out_channels attribute"
            raise AttributeError(msg) from e

        if necks is None:
            necks = []
        neck_list, channel_list = build_neck_list(necks, out_channels)

        # some decoders already include a head
        # for these, we pass the num_classes to them
        # others dont include a head
        # for those, we dont pass num_classes
        if not decoder_kwargs:
            decoder_kwargs, kwargs = extract_prefix_keys(kwargs, "decoder_")

        if not head_kwargs:
            head_kwargs, kwargs = extract_prefix_keys(kwargs, "head_")

        decoder, head_kwargs, decoder_includes_head = _get_decoder_and_head_kwargs(
            decoder, channel_list, decoder_kwargs, head_kwargs, num_classes=num_classes
        )

        if aux_decoders is None:
            _check_all_args_used(kwargs)
            return _build_appropriate_model(
                task,
                backbone,
                decoder,
                head_kwargs,
                patch_size=patch_size,
                padding=padding,
                necks=neck_list,
                decoder_includes_head=decoder_includes_head,
                rescale=rescale,
            )

        to_be_aux_decoders: list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] = []
        for aux_decoder in aux_decoders:
            args = aux_decoder.decoder_args if aux_decoder.decoder_args else {}
            aux_decoder_kwargs, args = extract_prefix_keys(args, "decoder_")
            aux_head_kwargs, args = extract_prefix_keys(args, "head_")
            aux_decoder_instance, aux_head_kwargs, aux_decoder_includes_head = _get_decoder_and_head_kwargs(
                aux_decoder.decoder, channel_list, aux_decoder_kwargs, aux_head_kwargs, num_classes=num_classes
            )
            to_be_aux_decoders.append(
                AuxiliaryHeadWithDecoderWithoutInstantiatedHead(aux_decoder.name, aux_decoder_instance, aux_head_kwargs)
            )
            _check_all_args_used(args)

        _check_all_args_used(kwargs)

        return _build_appropriate_model(
            task,
            backbone,
            decoder,
            head_kwargs,
            patch_size=patch_size,
            padding=padding,
            necks=neck_list,
            decoder_includes_head=decoder_includes_head,
            rescale=rescale,
            auxiliary_heads=to_be_aux_decoders,
        )


def _build_appropriate_model(
    task: str,
    backbone: nn.Module,
    decoder: nn.Module,
    head_kwargs: dict,
    patch_size: int | list | None,
    padding: str,
    decoder_includes_head: bool = False,
    necks: list[Neck] | None = None,
    rescale: bool = True,  # noqa: FBT001, FBT002
    auxiliary_heads: list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] | None = None,
):
    if necks:
        neck_module: nn.Module = nn.Sequential(*necks)
    else:
        neck_module = None
    if task in PIXEL_WISE_TASKS:
        return PixelWiseModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            patch_size=patch_size,
            padding=padding,
            decoder_includes_head=decoder_includes_head,
            neck=neck_module,
            rescale=rescale,
            auxiliary_heads=auxiliary_heads,
        )
    elif task in SCALAR_TASKS:
        return ScalarOutputModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            patch_size=patch_size,
            padding=padding,
            decoder_includes_head=decoder_includes_head,
            neck=neck_module,
            auxiliary_heads=auxiliary_heads,
        )
