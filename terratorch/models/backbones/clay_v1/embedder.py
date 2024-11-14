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

warnings.filterwarnings("ignore", category=UserWarning)


default_cfgs = generate_default_cfgs(
    {
        "clay_v1_base": {
            "hf_hub_id": "made-with-clay/Clay",
            "hf_hub_filename": "clay-v1-base.ckpt"
        }
    }
)


class Embedder(nn.Module):
    default_out_indices = (0,)  # Single out_indices for simplicity

    def __init__(self,
                 img_size=256,
                 num_frames=1,
                 ckpt_path=None,
                 bands=["blue", "green", "red", "nir", "swir16", "swir22"],
                 **kwargs):
        super().__init__()
        self.feature_info = []
        self.img_size = img_size
        self.num_frames = num_frames
        self.bands = bands

        if kwargs.get("datacuber", True) is not None:
            self.datacuber = Datacuber(bands=bands)
        else:
            self.datacuber = None
            
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
            )
        )

        # for use in features list. Single layer feature for simplicity
        self.feature_info.append({"num_chs": 768, "reduction": 1, "module": "clay_encoder"})

        # assuming this is used to fine tune a network on top of the embeddings

        if ckpt_path:
            self.load_clay_weights(ckpt_path)

    def load_clay_weights(self, ckpt_path):
        "Load the weights from the Clay model encoder."
        ckpt = torch.load(ckpt_path)
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
        return state_dict

    def forward_features(self, x):
        if self.datacuber is not None:
            datacube = self.datacuber(x)
        else:
            datacube = x
        embeddings = self.clay_encoder(datacube)

        # TODO: actually return features individually
        return [embeddings]

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
