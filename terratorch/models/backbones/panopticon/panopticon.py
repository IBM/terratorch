import torch
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
import json
import torch.nn as nn
from typing import List
import torchgeo.models.panopticon as pan 
from torchvision.models._api import Weights

with open('sensors_meta.json') as f:
    waves_list = json.load(f)

class PanopticonEncoderWrapper(nn.Module):

    """
    A wrapper for Panopticon models
    Attributes:
        model (Panopticon): The instantiated panopticon model
    Methods:
        forward(x: List[torch.Tensor], wavelengths: list[float]) -> torch.Tensor:
            Forward pass for embeddings with specified indices.
    """

    def __init__(self, model, wavelengths, n_blocks=12, return_class_token=False) -> None:
        """
        Args:
            model (Panopticon): The decoder module to be wrapped.
            wavelengths (list[float]): List of wavelengths for each channel.
            n_blocks (int): Number of blocks to return.
            return_class_token (bool): Whether to return the class token.
        """
        super().__init__()
        self.model = model
        self.wavelengths = wavelengths
        self.n_blocks = n_blocks
        self.out_indeces = list(torch.arange(0, n_blocks))
        self.out_channels = [self.model.embed_dim] * self.n_blocks
        self.return_class_token = return_class_token

    def forward(self, x: List[torch.Tensor], **kwargs) -> torch.Tensor:

        x_dict = dict(
            imgs = x, 
            chn_ids = torch.tensor(self.wavelengths).repeat(x.shape[0],1)  
        )

        outs = self.model.get_intermediate_layers(x_dict, n= self.out_indeces, return_class_token=self.return_class_token)

        if self.return_class_token:
            outs = list(outs)
            for i in range(len(outs)):
                outs[i] = torch.cat((outs[i][0],outs[i][1][:, None, :]), dim=1)
        
            outs = tuple(outs)
        
        return outs

def get_wavelenghts(model_bands: dict) -> list[float]:

    """Extract wavelength values for given spectral bands.
    
    Args:
        model_bands: dict of lists)
    
    Returns:
        List of corresponding wavelength values in nanometers
    """

    result = [
        waves_list[k][v[0]]
        for k, v in model_bands.items()
    ]

    return result
    


@TERRATORCH_BACKBONE_REGISTRY.register
def panopticon_vitb14(model_bands: dict, input_size: int = 224, pretrained: bool = False, weights: Weights | None = pan.Panopticon_Weights.VIT_BASE14, return_class_token: bool=True, **kwargs):

    """
    Initializes a Panopticon model and makes it TerraTorch compatible.

    Args:
    model_bands (list): A list of integers representing the spectral bands of the input data.
    input_size (int, optional): The size of the input images. Defaults to 224.
    pretrained (bool, optional): Whether to load pre-trained weights. Defaults to False.
    weights (Weights | None, optional): The pre-trained weights to use. Defaults to pan.Panopticon_Weights.VIT_BASE14.
    return_class_token (bool, optional): Whether to return the class token in the output. Defaults to True.
    **kwargs: Additional keyword arguments to pass to the Panopticon model constructor.

    Returns:
    PanopticonEncoderWrapper: An instance of PanopticonEncoderWrapper, which wraps the Panopticon model and provides additional functionality to enable the model on TerraTorch.
    """

    wavelengths = get_wavelenghts(model_bands)

    if pretrained == False:
        weights = None

    model = pan.panopticon_vitb14(img_size=input_size, weights=weights, **kwargs).model
    
    return PanopticonEncoderWrapper(model, wavelengths, n_blocks=12, return_class_token=return_class_token) 


