
import torch
import torch.nn as nn
from terratorch.cli_tools import SemanticSegmentationTask
from typing import Optional
from collections.abc import Iterable,Iterator

class MultiModalDataGenerator():
    def __init__(self,config: dict):
        print("DummyDataGenerator called")
        self.task_type= config["task_args"]["task"] 
    
    def _get_mm_fields_config(self) -> dict[str:str]:
        if self.task_type == "SemanticSegmentationTask": 
            return {
                "pixel_values": "image",
                "location_coords": "image"
            }

class DummyDataGenerator():
    def __init__(self,config: dict):
        print("DummyDataGenerator called")
        self.task_type= config["task_args"]["task"] 
    
    def get_dummy_mm_data(self) -> dict[str,torch.Tensor]:

        if self.task_type == "SemanticSegmentationTask":
            return {
            "pixel_values": torch.full((6, 512, 512), 1.0,
                                       dtype=torch.float16),
            "location_coords": torch.full((1, 2), 1.0, dtype=torch.float16),
        }
        else:
            raise Exception(f"task type {self.task_type} is not supported")
            

        
class InferenceRunner():
    
    def __init__(self, config: dict):
        

        self.task_type = config["task_args"]["task"]

        if self.task_type == "SemanticSegmentationTask":

            self.task = SemanticSegmentationTask(
                config["model_args"],
                config["task_args"]["model_factory"],
                loss=config["task_args"]["loss"],
                lr=config["task_args"]["lr"],
                ignore_index=config["task_args"]["ignore_index"],
                optimizer=config["task_args"]["optimizer"],
                optimizer_hparams=config["optimizer_params"],
                scheduler=config["task_args"]["scheduler"],
                scheduler_hparams=config["scheduler_params"],
                plot_on_val=config["task_args"]["plot_on_val"],
                freeze_decoder=config["task_args"]["freeze_decoder"],
                freeze_backbone=config["task_args"]["freeze_backbone"],
            )

        else:
            raise ValueError(
                "Unsupported task. "
                "Only SemanticSegmentationTask is supported for now "
                "by PrithviGeospatialMAE.")
    
    def _parse_and_validate_multimodal_data(
        self,**kwargs:object) -> tuple[torch.Tensor, Optional[torch.Tensor]] :


        if self.task_type == "SemanticSegmentationTask":

            pixel_values = kwargs.pop("pixel_values", None)
            if not isinstance(pixel_values, torch.Tensor):
                raise ValueError(f"Incorrect type of pixel_values. "
                                f"Got type: {type(pixel_values)}")

            location_coords = kwargs.pop("location_coords", None)
            if not isinstance(location_coords, torch.Tensor):
                raise ValueError(f"Incorrect type of location_coords. "
                                f"Got type: {type(location_coords)}")
            location_coords = torch.unbind(location_coords, dim=0)[0]
            if location_coords.shape == torch.Size([0]):
                location_coords = None

            return pixel_values, location_coords
        else:
            raise Exception(f"task type {self.task_type} is not supported")
    
    def forward(self,**kwargs: object):
        
        pixel_values, location_coords = (
            self._parse_and_validate_multimodal_data(**kwargs))
        model_output = self.task.model(pixel_values,
                                  location_coords=location_coords)
        return model_output
    