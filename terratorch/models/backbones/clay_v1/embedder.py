import re
import warnings

import numpy as np
import torch
from torch import nn, Tensor
import torch
from timm.models import FeatureInfo
from timm.models._builder import build_model_with_cfg
from timm.models._registry import generate_default_cfgs, register_model

from terratorch.models.backbones.clay_v1.modules import EmbeddingEncoder, Datacuber


default_cfgs = generate_default_cfgs(
    {
        "clay_v1_base": {
            "hf_hub_id": "made-with-clay/Clay",
            "hf_hub_filename": "v1/clay-v1-base.ckpt",
        }
    }
)


class Embedder(nn.Module):
    default_out_indices = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

    def __init__(
        self,
        img_size=256,
        num_frames=1,
        ckpt_path=None,
        bands=["blue", "green", "red", "nir", "swir16", "swir22"],
        out_indices: tuple[int] = default_out_indices,
        vpt: bool = False,
        vpt_n_tokens: int | None = None,
        vpt_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.feature_info = []
        self.img_size = img_size
        self.num_frames = num_frames
        self.bands = bands
        self.out_indices = out_indices

        self.datacuber = Datacuber(bands=bands)

        # TODO: add support for various clay versions
        self.clay_encoder = (
            EmbeddingEncoder(  # Default parameters for the Clay base model
                img_size=img_size,
                patch_size=8,
                dim=768,
                depth=12,
                heads=12,
                dim_head=64,
                mlp_ratio=4.0,
                vpt=vpt,
                vpt_n_tokens=vpt_n_tokens,
                vpt_dropout=vpt_dropout,
            )
        )

        # for use in features list.
        for i in range(12):
            self.feature_info.append({"num_chs": 768, "reduction": 1, "module": f"blocks.{i}"})

        # assuming this is used to fine tune a network on top of the embeddings

        if ckpt_path:
            self.load_clay_weights(ckpt_path)

    def load_clay_weights(self, ckpt_path):
        "Load the weights from the Clay model encoder."
        ckpt = torch.load(ckpt_path, weights_only=True)
        state_dict = ckpt.get("state_dict")
        state_dict = {
            re.sub(r"^model\.encoder\.", "", name): param
            for name, param in state_dict.items()
            if name.startswith("model.encoder")
        }

        with torch.no_grad():
            for name, param in self.clay_encoder.named_parameters():
                if name in state_dict and param.size() == state_dict[name].size():
                    param.data.copy_(state_dict[name])  # Copy the weights
                else:
                    print(
                        f"No matching parameter for {name} with size {param.size()}")

        for param in self.clay_encoder.parameters():
            param.requires_grad = False

        self.clay_encoder.eval()

    @staticmethod
    def transform_state_dict(state_dict, model):
        state_dict = state_dict.get("state_dict")
        state_dict = {
            re.sub(r"^model\.encoder\.", "clay_encoder.", name): param
            for name, param in state_dict.items()
            if name.startswith("model.encoder")
        }
        for k, v in model.state_dict().items():
            if "vpt_prompt_embeddings" in k:
                state_dict[k] = v
        return state_dict

    def forward_features(
        self,
        x: torch.Tensor,
        time: torch.Tensor | None = None,
        latlon: torch.Tensor | None = None,
        waves: torch.Tensor | None = None,
        gsd: float | None = None,
    ):
        datacube = self.datacuber(x=x, time=time, latlon=latlon, waves=waves, gsd=gsd)
        embeddings = self.clay_encoder(datacube)

        return [embeddings[i] for i in self.out_indices]

    def fake_datacube(self):
        "Generate a fake datacube for model export."
        dummy_datacube = {
            "pixels": torch.randn(2, 3, self.img_size, self.img_size),
            "time": torch.randn(2, 4),
            "latlon": torch.randn(2, 4),
            "waves": torch.randn(3),
            "gsd": torch.randn(1),
        }
        dummy_datacube = {k: v
                          for k, v in dummy_datacube.items()}
        return dummy_datacube

    def prepare_features_for_image_model(self, features: list[Tensor]) -> list[Tensor]:
        x_no_token = features[-1][:, 1:, :]
        encoded = x_no_token.permute(0, 2, 1).reshape(
            x_no_token.shape[0],
            -1,
            int(np.sqrt(x_no_token.shape[1] // self.num_frames)),
            int(np.sqrt(x_no_token.shape[1] // self.num_frames)),
        )
        
        # return as list for features list compatibility
        return [encoded]

    def freeze(self):
        for n, param in self.named_parameters():
            if "vpt_prompt_embeddings" not in n:
                param.requires_grad_(False)


def _make_clay(
    variant: str,
    pretrained: bool,
    **kwargs
):
    encoder_only = kwargs.pop("features_only", False)
    model = build_model_with_cfg(
        model_cls=Embedder,
        variant=variant,
        pretrained=pretrained,
        pretrained_strict=True,
        pretrained_filter_fn=Embedder.transform_state_dict,
        **kwargs,
    )
    if encoder_only:
        out_indices = kwargs.pop("out_indices", model.default_out_indices)
        model.feature_info = FeatureInfo(model.feature_info, out_indices)
        model.model_bands = kwargs.get("model_bands")

        # TODO: split features according to typical TIMM outputs
        model.forward = model.forward_features
        model.pretrained_bands = kwargs.get("pretrained_bands")
    return model


@register_model
def clay_v1_base(
    pretrained: bool = False,
    **kwargs,
) -> Embedder:
    return _make_clay(
        "clay_v1_base",
        pretrained=pretrained,
        **kwargs
    )
