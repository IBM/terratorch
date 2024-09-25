# Copyright contributors to the Terratorch project

from collections.abc import Callable

import torch
from segmentation_models_pytorch.base import SegmentationModel
from torch import nn

from terratorch.models.heads import ClassificationHead
from terratorch.models.model import Model, ModelOutput


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad_(False)


class ScalarOutputModel(Model, SegmentationModel):
    """Model that encapsulates encoder and decoder and heads for a scalar output
    Expects decoder to have a "forward_features" method, an embed_dims property
    and optionally a "prepare_features_for_image_model" method.
    """

    def __init__(
        self,
        task: str,
        encoder: nn.Module,
        decoder: nn.Module,
        head_kwargs: dict,
        auxiliary_heads: dict[str, nn.Module] | None = None,
        prepare_features_for_image_model: Callable | None = None,
    ) -> None:
        """Constructor

        Args:
            task (str): Task to be performed. Must be "classification".
            encoder (nn.Module): Encoder to be used
            decoder (nn.Module): Decoder to be used
            head_kwargs (dict): Arguments to be passed at instantiation of the head.
            auxiliary_heads (dict[str, nn.Module] | None, optional): Names mapped to auxiliary heads. Defaults to None.
            prepare_features_for_image_model (Callable | None, optional): Function applied to encoder outputs.
                Defaults to None.
        """
        super().__init__()
        self.task = task
        self.encoder = encoder
        self.decoder = decoder
        self.head = self._get_head(task, decoder.output_embed_dim, head_kwargs)

        if auxiliary_heads is not None:
            aux_heads = {}
            for aux_head_to_be_instantiated in auxiliary_heads:
                aux_head: nn.Module = self._get_head(
                    task, aux_head_to_be_instantiated.decoder.output_embed_dim, **head_kwargs
                )
                aux_head = nn.Sequential(aux_head_to_be_instantiated.decoder, aux_head)
                aux_heads[aux_head_to_be_instantiated.name] = aux_head
        else:
            aux_heads = {}
        self.aux_heads = nn.ModuleDict(aux_heads)

        self.prepare_features_for_image_model = prepare_features_for_image_model

    def freeze_encoder(self):
        freeze_module(self.encoder)

    def freeze_decoder(self):
        freeze_module(self.encoder)
        freeze_module(self.head)

    # TODO: do this properly
    def check_input_shape(self, x: torch.Tensor) -> bool:  # noqa: ARG002
        return True

    def forward(self, x: torch.Tensor, **kwargs) -> ModelOutput:
        """Sequentially pass `x` through model`s encoder, decoder and heads"""

        self.check_input_shape(x)
        features = self.encoder(x, **kwargs)

        # some models need their features reshaped

        if self.prepare_features_for_image_model:
            prepare = self.prepare_features_for_image_model
        else:
            prepare = getattr(self.encoder, "prepare_features_for_image_model", lambda x: x)
        features = prepare(features)

        decoder_output = self.decoder([f.clone() for f in features])
        mask = self.head(decoder_output)
        aux_outputs = {}
        for name, decoder in self.aux_heads.items():
            aux_output = decoder([f.clone() for f in features])
            aux_outputs[name] = aux_output
        return ModelOutput(output=mask, auxiliary_heads=aux_outputs)

    def _get_head(self, task: str, input_embed_dim: int, head_kwargs: dict):
        if task == "classification":
            if "num_classes" not in head_kwargs:
                msg = "num_classes must be defined for classification task"
                raise Exception(msg)
            return ClassificationHead(input_embed_dim, **head_kwargs)
        msg = "Task must be classification."
        raise Exception(msg)
